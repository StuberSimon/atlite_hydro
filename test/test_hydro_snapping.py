# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

import logging
import os
import sys
from types import SimpleNamespace

# point pyproj at the conda PROJ database if it is not found automatically
_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.exists(os.path.join(_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _proj)
    import pyproj

    pyproj.datadir.set_data_dir(_proj)

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString

from atlite.hydro import _hydro_from_discharge, normalize_river, snap_plants_to_river


def _proj_ok():
    try:
        gpd.GeoSeries([], crs="EPSG:4326").to_crs("EPSG:3857")
        return True
    except Exception:
        return False


requires_proj = pytest.mark.skipif(
    not _proj_ok(), reason="pyproj PROJ database not available"
)

# 11x11 grid at 0.05 deg, coords rounded to 5 decimals (GLOFAS convention).
XS = np.round(10.00 + 0.05 * np.arange(11), 5)
YS = np.round(50.00 + 0.05 * np.arange(11), 5)


def _uparea(values):
    """Wrap a (ny, nx) array as a km2 uparea DataArray on the test grid."""
    return xr.DataArray(
        values,
        coords={"y": YS, "x": XS},
        dims=("y", "x"),
        name="uparea",
        attrs={"units": "km2"},
    )


@pytest.fixture
def main_river():
    """Single N-S river at ix=5, uparea growing downstream (south)."""
    a = np.full((11, 11), np.nan)
    a[:, 5] = 5000 - 400 * np.arange(11)  # iy=0 -> 5000, iy=10 -> 1000
    return _uparea(a)


@pytest.fixture
def river_with_tributary():
    """Main river ix=5 (~4000) plus a small tributary ix=3 (300)."""
    a = np.full((11, 11), np.nan)
    a[:, 5] = 4000.0
    a[:, 3] = 300.0
    return _uparea(a)


def test_misaligned_plant_snaps_onto_river(main_river):
    # plant one cell west of the river
    plants = pd.DataFrame({"lon": [10.20], "lat": [50.25]}, index=["p"])
    out = snap_plants_to_river(plants, main_river)
    assert out.loc["p", "x_snapped"] == pytest.approx(10.25)  # on the river column
    assert out.loc["p", "snap_status"] == "ok"
    # max-uparea over the window picks the largest in-window river cell
    assert out.loc["p", "uparea_snapped"] == pytest.approx(4200.0)
    assert out.loc["p", "snap_distance"] > 0


def test_fork_max_vs_area(river_with_tributary):
    # plant sits on the tributary, main stem is within the search window
    plants = pd.DataFrame({"lon": [10.15], "lat": [50.25]}, index=["p"])

    max_out = snap_plants_to_river(plants, river_with_tributary, method="max")
    # the known failure: max latches onto the larger river
    assert max_out.loc["p", "x_snapped"] == pytest.approx(10.25)

    plants_area = plants.assign(catchment_area=[300.0])
    area_out = snap_plants_to_river(plants_area, river_with_tributary, method="area")
    # area matching recovers the correct tributary branch
    assert area_out.loc["p", "x_snapped"] == pytest.approx(10.15)
    assert area_out.loc["p", "snap_status"] == "ok"


def test_area_tolerance_fallback(river_with_tributary):
    plants = pd.DataFrame(
        {"lon": [10.15], "lat": [50.25], "catchment_area": [50.0]}, index=["p"]
    )
    out = snap_plants_to_river(plants, river_with_tributary, method="area")
    # no cell within accordance -> fall back to max choice
    assert out.loc["p", "snap_status"] == "fallback_max"
    assert out.loc["p", "x_snapped"] == pytest.approx(10.25)


def test_mixed_table_rowwise_fallback(river_with_tributary):
    plants = pd.DataFrame(
        {
            "lon": [10.15, 10.15],
            "lat": [50.25, 50.25],
            "catchment_area": [300.0, np.nan],
        },
        index=["good", "nan"],
    )
    out = snap_plants_to_river(plants, river_with_tributary, method="area")
    assert out.loc["good", "snap_status"] == "ok"
    assert out.loc["good", "x_snapped"] == pytest.approx(10.15)
    assert out.loc["nan", "snap_status"] == "fallback_max"
    assert out.loc["nan", "x_snapped"] == pytest.approx(10.25)


def test_ambiguity_flag(caplog):
    # two parallel rivers of equal size separated by NaN land
    a = np.full((11, 11), np.nan)
    a[:, 2] = 1000.0
    a[:, 8] = 1000.0
    uparea = _uparea(a)
    plants = pd.DataFrame({"lon": [10.25], "lat": [50.25]}, index=["p"])
    with caplog.at_level(logging.WARNING):
        out = snap_plants_to_river(plants, uparea, radius=5)
    assert out.loc["p", "snap_status"] == "ambiguous"
    assert "ambiguous" in caplog.text


def test_single_river_not_ambiguous(main_river):
    # a lone river must not flag itself ambiguous via its own up/downstream cells
    plants = pd.DataFrame({"lon": [10.25], "lat": [50.25]}, index=["p"])
    out = snap_plants_to_river(plants, main_river)
    assert out.loc["p", "snap_status"] == "ok"


def test_no_river_status():
    a = np.full((11, 11), np.nan)  # all ocean
    uparea = _uparea(a)
    plants = pd.DataFrame({"lon": [10.25], "lat": [50.25]}, index=["p"])
    out = snap_plants_to_river(plants, uparea)
    assert out.loc["p", "snap_status"] == "no_river"
    # keeps the nearest cell, discharge would be NaN there
    assert out.loc["p", "x_snapped"] == pytest.approx(10.25)


def test_edge_plant_does_not_raise(main_river):
    # plant in the grid corner; window clipping must not error
    plants = pd.DataFrame({"lon": [10.00], "lat": [50.00]}, index=["p"])
    out = snap_plants_to_river(plants, main_river)
    assert out.loc["p", "snap_status"] in {"ok", "no_river"}


def test_plant_outside_domain_raises(main_river):
    plants = pd.DataFrame({"lon": [20.0], "lat": [50.25]}, index=["p"])
    with pytest.raises(ValueError, match="outside the uparea domain"):
        snap_plants_to_river(plants, main_river)


def test_method_area_requires_column(main_river):
    plants = pd.DataFrame({"lon": [10.25], "lat": [50.25]}, index=["p"])
    with pytest.raises(ValueError, match="catchment_area"):
        snap_plants_to_river(plants, main_river, method="area")


def test_units_m2_converted():
    a = np.full((11, 11), np.nan)
    a[:, 5] = 3.0e9  # m2 -> 3000 km2
    uparea = _uparea(a)
    uparea.attrs["units"] = "m2"
    plants = pd.DataFrame({"lon": [10.25], "lat": [50.25]}, index=["p"])
    out = snap_plants_to_river(plants, uparea)
    assert out.loc["p", "uparea_snapped"] == pytest.approx(3000.0)


# --- integration with _hydro_from_discharge ---------------------------------


def _fake_cutout(discharge):
    ds = discharge.to_dataset(name="discharge")
    return SimpleNamespace(data=ds, coords=ds.coords)


def _discharge_grid():
    """discharge[t, y, x] encodes the cell identity ix*100 + iy, constant in t."""
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    ix = np.arange(11)
    iy = np.arange(11)
    code = (ix[None, :] * 100 + iy[:, None]).astype(float)  # (ny, nx)
    values = np.broadcast_to(code, (2, 11, 11))
    return xr.DataArray(
        values,
        coords={"time": time, "y": YS, "x": XS},
        dims=("time", "y", "x"),
        name="discharge",
    )


def test_hydro_uses_snapped_columns():
    cutout = _fake_cutout(_discharge_grid())
    # snapped onto cell ix=5, iy=3 -> code 503
    plants = pd.DataFrame(
        {
            "lon": [10.20],
            "lat": [50.20],
            "x_snapped": [XS[5]],
            "y_snapped": [YS[3]],
        },
        index=["p"],
    )
    inflow = _hydro_from_discharge(cutout, plants)
    assert float(inflow.sel(plant="p").isel(time=0)) == pytest.approx(503.0)


def test_hydro_without_snapped_uses_nearest(caplog):
    cutout = _fake_cutout(_discharge_grid())
    # nearest cell to (10.24, 50.06) is ix=5, iy=1 -> code 501
    plants = pd.DataFrame({"lon": [10.24], "lat": [50.06]}, index=["p"])
    with caplog.at_level(logging.INFO):
        inflow = _hydro_from_discharge(cutout, plants)
    assert float(inflow.sel(plant="p").isel(time=0)) == pytest.approx(501.0)
    assert "snap_plants_to_river" in caplog.text


def test_hydro_rejects_misaligned_snapped_coords():
    cutout = _fake_cutout(_discharge_grid())
    # x_snapped beyond the cutout grid (e.g. from a mismatched uparea grid)
    plants = pd.DataFrame(
        {"lon": [10.20], "lat": [50.20], "x_snapped": [10.65], "y_snapped": [50.20]},
        index=["p"],
    )
    with pytest.raises(ValueError, match="not aligned with the cutout grid"):
        _hydro_from_discharge(cutout, plants)


# --- additive extension: passthrough + river-aware --------------------------


def test_normalize_river_aliases():
    assert normalize_river("Danube River") == "donau"
    assert normalize_river("Rhein") == "rhein"
    assert normalize_river("Innkanal") == "inn"
    assert normalize_river("  ") is None
    assert normalize_river(None) is None
    assert normalize_river("Foo", {"foo": "bar"}) == "bar"


def test_no_new_args_identical_output(river_with_tributary):
    # additive params default to a no-op: the existing snapped columns are unchanged
    plants = pd.DataFrame(
        {"lon": [10.15], "lat": [50.25], "catchment_area": [300.0]}, index=["p"]
    )
    out = snap_plants_to_river(plants, river_with_tributary, method="area")
    assert out.loc["p", "x_snapped"] == pytest.approx(10.15)
    assert out.loc["p", "snap_status"] == "ok"
    # new column is present but inactive
    assert "matched_river" in out.columns
    assert pd.isna(out.loc["p", "matched_river"])


def test_passthrough_technologies(river_with_tributary):
    plants = pd.DataFrame(
        {
            "lon": [10.15, 10.15],
            "lat": [50.25, 50.25],
            "technology": ["Run-Of-River", "Pumped Storage"],
            "catchment_area": [300.0, 300.0],
        },
        index=["ror", "phs"],
    )
    out = snap_plants_to_river(
        plants,
        river_with_tributary,
        method="area",
        passthrough_technologies=["Pumped Storage"],
    )
    assert out.loc["phs", "snap_status"] == "passthrough"
    assert np.isnan(out.loc["phs", "x_snapped"])
    assert np.isnan(out.loc["phs", "y_snapped"])
    assert np.isnan(out.loc["phs", "snap_distance"])
    # the non-passthrough plant is snapped normally
    assert out.loc["ror", "snap_status"] == "ok"
    assert out.loc["ror", "x_snapped"] == pytest.approx(10.15)


def test_hydro_drops_passthrough_plants():
    cutout = _fake_cutout(_discharge_grid())
    plants = pd.DataFrame(
        {
            "lon": [10.20, 10.20],
            "lat": [50.20, 50.20],
            "x_snapped": [XS[5], np.nan],  # phs has no snapped cell
            "y_snapped": [YS[3], np.nan],
        },
        index=["ror", "phs"],
    )
    inflow = _hydro_from_discharge(cutout, plants)
    assert list(inflow.plant.values) == ["ror"]
    assert float(inflow.sel(plant="ror").isel(time=0)) == pytest.approx(503.0)


@pytest.fixture
def two_named_rivers():
    """Big 'Salzach' (ix=5, 6000) and smaller 'Alz' (ix=3, 2000) with named lines."""
    a = np.full((11, 11), np.nan)
    a[:, 5] = 6000.0
    a[:, 3] = 2000.0
    uparea = _uparea(a)
    rivers = gpd.GeoDataFrame(
        {"name": ["Alz", "Salzach"]},
        geometry=[
            LineString([(XS[3], YS[0]), (XS[3], YS[-1])]),
            LineString([(XS[5], YS[0]), (XS[5], YS[-1])]),
        ],
        crs="EPSG:4326",
    )
    return uparea, rivers


@requires_proj
def test_river_aware_confluence(two_named_rivers):
    uparea, rivers = two_named_rivers
    # plant on the smaller Alz; max-uparea alone would grab the bigger Salzach
    plants = pd.DataFrame(
        {"lon": [10.15], "lat": [50.25], "river": ["Alz"]}, index=["p"]
    )
    base = snap_plants_to_river(plants, uparea, method="max")
    assert base.loc["p", "x_snapped"] == pytest.approx(10.25)  # wrong (Salzach)

    out = snap_plants_to_river(
        plants, uparea, method="max", rivers=rivers, river_col="river"
    )
    assert out.loc["p", "snap_status"] == "river_match"
    assert out.loc["p", "x_snapped"] == pytest.approx(10.15)  # correct (Alz)
    assert out.loc["p", "uparea_snapped"] == pytest.approx(2000.0)
    assert out.loc["p", "matched_river"] == "alz"


@requires_proj
def test_river_aware_graceful_skip_missing_file(two_named_rivers, caplog):
    uparea, _ = two_named_rivers
    plants = pd.DataFrame(
        {"lon": [10.15], "lat": [50.25], "river": ["Alz"]}, index=["p"]
    )
    with caplog.at_level(logging.WARNING):
        out = snap_plants_to_river(
            plants,
            uparea,
            method="max",
            rivers="/nonexistent/does_not_exist.gpkg",
            river_col="river",
        )
    # falls back to the max snap without raising
    assert out.loc["p", "snap_status"] == "ok"
    assert out.loc["p", "x_snapped"] == pytest.approx(10.25)
    assert pd.isna(out.loc["p", "matched_river"])
    assert "river-aware snapping skipped" in caplog.text

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module involving hydro operations in atlite.
"""

import logging
from collections import namedtuple
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
from shapely.geometry import Point
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between two (broadcastable) lon/lat points."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


Basins = namedtuple("Basins", ["plants", "meta", "shapes"])


def find_basin(shapes, lon, lat):
    hids = shapes.index[shapes.intersects(Point(lon, lat))]
    if len(hids) > 1:
        logger.warning(
            f"The point ({lon}, {lat}) is in several basins: {hids}. "
            "Assuming the first one."
        )
    return hids[0]


def find_upstream_basins(meta, hid):
    hids = [hid]
    i = 0
    while i < len(hids):
        hids.extend(meta.index[meta["NEXT_DOWN"] == hids[i]])
        i += 1
    return hids


def determine_basins(plants, hydrobasins, show_progress=False):
    if isinstance(hydrobasins, str):
        hydrobasins = gpd.read_file(hydrobasins)

    assert isinstance(hydrobasins, gpd.GeoDataFrame), (
        "hydrobasins should be passed as a filename or a GeoDataFrame, "
        f"but received `type(hydrobasins) = {type(hydrobasins)}`"
    )

    missing_columns = pd.Index(
        ["HYBAS_ID", "DIST_MAIN", "NEXT_DOWN", "geometry"]
    ).difference(hydrobasins.columns)
    assert missing_columns.empty, (
        "Couldn't find the column(s) {} in the hydrobasins dataset.".format(
            ", ".join(missing_columns)
        )
    )

    hydrobasins = hydrobasins.set_index("HYBAS_ID")

    meta = hydrobasins[hydrobasins.columns.difference(("geometry",))]
    shapes = hydrobasins["geometry"]

    plant_basins = []
    for p in tqdm(
        plants.itertuples(),
        disable=not show_progress,
        desc="Determine upstream basins per plant",
    ):
        hid = find_basin(shapes, p.lon, p.lat)
        plant_basins.append((hid, find_upstream_basins(meta, hid)))
    plant_basins = pd.DataFrame(
        plant_basins, columns=["hid", "upstream"], index=plants.index
    )

    unique_basins = pd.Index(plant_basins["upstream"].sum()).unique().rename("hid")
    return Basins(plant_basins, meta.loc[unique_basins], shapes.loc[unique_basins])


def shift_and_aggregate_runoff_for_plants(
    basins, runoff, flowspeed=1, show_progress=False
):
    inflow = xr.DataArray(
        np.zeros((len(basins.plants), runoff.indexes["time"].size)),
        [("plant", basins.plants.index), runoff.coords["time"]],
    )

    for ppl in tqdm(
        basins.plants.itertuples(),
        disable=not show_progress,
        desc="Shift and aggregate runoff by plant",
    ):
        inflow_plant = inflow.loc[dict(plant=ppl.Index)]
        distances = (
            basins.meta.loc[ppl.upstream, "DIST_MAIN"]
            - basins.meta.at[ppl.hid, "DIST_MAIN"]
        )
        nhours = (distances / (flowspeed * 3.6) + 0.5).astype(int)

        for b in ppl.upstream:
            inflow_plant += runoff.sel(hid=b).roll(time=nhours.at[b])

    return inflow


def _hydro_from_runoff(
    cutout,
    plants,
    hydrobasins,
    flowspeed=1,
    weight_with_height=False,
    show_progress=False,
    **kwargs,
):
    """
    Compute inflow time-series for `plants` by aggregating over catchment
    basins from `hydrobasins` (ERA5 runoff-based computation).

    Parameters
    ----------
    plants : pd.DataFrame
        Run-of-river plants or dams with lon, lat columns.
    hydrobasins : str|gpd.GeoDataFrame
        Filename or GeoDataFrame of one level of the HydroBASINS dataset.
    flowspeed : float
        Average speed of water flows to estimate the water travel time from
        basin to plant (default: 1 m/s).
    weight_with_height : bool
        Whether surface runoff should be weighted by potential height (probably
        better for coarser resolution).
    show_progress : bool
        Whether to display progressbars.

    References
    ----------
    [1] Liu, Hailiang, et al. "A validated high-resolution hydro power
    time-series model for energy systems analysis." arXiv preprint
    arXiv:1901.08476 (2019).

    [2] Lehner, B., Grill G. (2013): Global river hydrography and network
    routing: baseline data and new approaches to study the world’s large river
    systems. Hydrological Processes, 27(15): 2171–2186. Data is available at
    www.hydrosheds.org.

    """
    basins = determine_basins(plants, hydrobasins, show_progress=show_progress)

    matrix = cutout.indicatormatrix(basins.shapes)
    # compute the average surface runoff in each basin
    # Fix NaN and Inf values to 0.0 to avoid numerical issues
    matrix_normalized = np.nan_to_num(
        matrix / matrix.sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0
    )
    runoff = cutout.runoff(
        matrix=matrix_normalized,
        index=basins.shapes.index,
        weight_with_height=weight_with_height,
        show_progress=show_progress,
        **kwargs,
    )
    # The hydrological parameters are in units of "m of water per day" and so
    # they should be multiplied by 1000 and the basin area to convert to m3
    # d-1 = m3 h-1 / 24
    runoff *= xr.DataArray(basins.shapes.to_crs(dict(proj="cea")).area)

    return shift_and_aggregate_runoff_for_plants(
        basins, runoff, flowspeed, show_progress
    )


def _load_uparea(uparea):
    """Normalize the `uparea` input of `snap_plants_to_river` to a km2 DataArray."""
    if isinstance(uparea, (str, Path)):
        # lazy import: glofas.py imports cdsapi at module level
        from atlite.datasets.glofas import retrieve_uparea

        return retrieve_uparea(uparea)  # already km2, coords cleaned
    if isinstance(uparea, xr.Dataset):
        uparea = uparea["uparea"]
    if uparea.attrs.get("units", "").lower() in ("m2", "m**2", "m^2"):
        uparea = uparea / 1e6
        uparea.attrs["units"] = "km2"
    return uparea


# river-name normalization: plant tables use mixed English/German ("Danube
# River", "Rhein", "Innkanal"); OSM uses local names ("Donau", "Rhein", "Inn").
# Strip suffixes and map the cross-language names so the two agree.
_RIVER_ALIASES = {
    "danube": "donau",
    "rhine": "rhein",
    "meuse": "maas",
    "moselle": "mosel",
    "the danube": "donau",
}


def normalize_river(name, aliases=None):
    """Lowercase, strip ' river'/' fluss'/' canal' suffixes and 'kanal', apply aliases."""
    if not isinstance(name, str) or not name.strip():
        return None
    s = name.lower().strip()
    for suf in (" river", " fluss", " canal"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = s.replace("kanal", "").strip()
    table = {**_RIVER_ALIASES, **(aliases or {})}
    return table.get(s, s)


def _load_waterways(rivers, river_name_col, river_aliases):
    """Load named waterway lines (GeoDataFrame or path) in Web-Mercator with a
    normalized `rname` column for metric nearest-distance joins."""
    if isinstance(rivers, (str, Path)):
        rivers = gpd.read_file(rivers)
    wj = rivers[rivers[river_name_col].notna()].copy()
    wj["rname"] = wj[river_name_col].map(lambda n: normalize_river(n, river_aliases))
    return wj.to_crs("EPSG:3857")


def _snap_river_aware(
    plants,
    snapped,
    sub,
    rivers,
    river_col,
    river_name_col,
    river_aliases,
    radius,
    match_dist_km,
    onstream_dist_km,
):
    """Refine the area/max snap by moving each plant onto the max-uparea cell that
    lies on its named river (resolves confluences). Never raises: on any failure it
    logs a warning and returns the unchanged area-snap result."""
    try:
        wj = _load_waterways(rivers, river_name_col, river_aliases)
    except Exception as e:
        logger.warning(f"river-aware snapping skipped (could not load rivers): {e}")
        return snapped

    dx = float(sub.x[1] - sub.x[0])
    out = snapped.copy()
    status = out["snap_status"].astype(object).copy()
    matched = out["matched_river"].astype(object).copy()
    river_names = plants[river_col]
    n_match = 0
    for idx in plants.index:
        rn = normalize_river(river_names.at[idx], river_aliases)
        if rn is None:
            continue
        lon = float(plants.at[idx, "lon"])
        lat = float(plants.at[idx, "lat"])
        win = sub.sel(
            x=slice(lon - (radius + 0.5) * dx, lon + (radius + 0.5) * dx),
            y=slice(lat - (radius + 0.5) * dx, lat + (radius + 0.5) * dx),
        )
        X, Y = np.meshgrid(win.x.values, win.y.values)
        A = win.values.ravel()
        lon_c, lat_c = X.ravel(), Y.ravel()
        ok = np.isfinite(A)
        if not ok.any():
            continue
        # cell centres carry their lon/lat as columns so they survive the join
        cells = gpd.GeoDataFrame(
            {"uparea": A[ok], "clon": lon_c[ok], "clat": lat_c[ok]},
            geometry=gpd.points_from_xy(lon_c[ok], lat_c[ok]),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        try:
            lab = gpd.sjoin_nearest(
                cells,
                wj[[river_name_col, "rname", "geometry"]],
                max_distance=match_dist_km * 1000,
                distance_col="d",
            )
        except Exception as e:
            logger.warning(f"river-aware snapping skipped (spatial join failed): {e}")
            return snapped
        hit = lab[lab["rname"] == rn]
        if hit.empty:
            continue
        # prefer cells the named line runs through (guards against a mislabelled
        # bigger river at a confluence), then take the main stem (max uparea)
        onstream = hit[hit["d"] <= onstream_dist_km * 1000]
        use = onstream if not onstream.empty else hit
        best = use.loc[use["uparea"].idxmax()]
        out.at[idx, "x_snapped"] = float(best["clon"])
        out.at[idx, "y_snapped"] = float(best["clat"])
        out.at[idx, "uparea_snapped"] = float(best["uparea"])
        out.at[idx, "snap_distance"] = float(
            _haversine_km(lon, lat, float(best["clon"]), float(best["clat"]))
        )
        status.at[idx] = "river_match"
        matched.at[idx] = rn
        n_match += 1
    out["snap_status"] = status
    out["matched_river"] = matched
    logger.info(f"river-aware snap: matched {n_match}/{len(plants)} plants to their river")
    return out


def _snap_core(
    plants,
    uparea,
    method,
    radius,
    area_col,
    min_accordance,
    distance_weight,
    min_uparea,
    ambiguity_ratio,
):
    """Core max/area uparea snap. Returns (result, sub) where `sub` is the loaded
    uparea window for reuse by the river-aware refinement."""
    uparea = _load_uparea(uparea).transpose("y", "x")
    xfull = uparea.x.values
    yfull = uparea.y.values
    dx = (xfull[-1] - xfull[0]) / (xfull.size - 1)
    dy = (yfull[-1] - yfull[0]) / (yfull.size - 1)

    lon = plants["lon"].to_numpy(dtype=float)
    lat = plants["lat"].to_numpy(dtype=float)

    outside = (
        (lon < xfull[0] - dx / 2)
        | (lon > xfull[-1] + dx / 2)
        | (lat < yfull[0] - dy / 2)
        | (lat > yfull[-1] + dy / 2)
    )
    if outside.any():
        raise ValueError(
            f"Plants outside the uparea domain: {list(plants.index[outside])}"
        )

    # slice to the plants' bounding box (+ margin) and load into memory
    margin_x = (radius + 2) * dx
    margin_y = (radius + 2) * dy
    sub = uparea.sel(
        x=slice(lon.min() - margin_x, lon.max() + margin_x),
        y=slice(lat.min() - margin_y, lat.max() + margin_y),
    ).load()
    xs = sub.x.values
    ys = sub.y.values
    nx, ny = xs.size, ys.size
    A = sub.values  # (ny, nx)

    # nearest cell per plant on the regular grid
    ix = np.clip(np.rint((lon - xs[0]) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((lat - ys[0]) / dy).astype(int), 0, ny - 1)

    P = len(plants)
    W = 2 * radius + 1
    offs = np.arange(-radius, radius + 1)
    cand_iy = np.clip(iy[:, None, None] + offs[None, :, None], 0, ny - 1)  # (P,W,1)
    cand_ix = np.clip(ix[:, None, None] + offs[None, None, :], 0, nx - 1)  # (P,1,W)
    cand_iy = np.broadcast_to(cand_iy, (P, W, W))
    cand_ix = np.broadcast_to(cand_ix, (P, W, W))

    A_win = A[cand_iy, cand_ix]  # (P,W,W)
    cx = xs[cand_ix]
    cy = ys[cand_iy]
    d = _haversine_km(lon[:, None, None], lat[:, None, None], cx, cy)
    d_max = np.maximum(d.max(axis=(1, 2), keepdims=True), 1e-9)

    valid = ~np.isnan(A_win) & (A_win >= min_uparea)
    any_valid = valid.reshape(P, -1).any(axis=1)

    # max-uparea choice
    A_masked = np.where(valid, A_win, -np.inf)
    max_flat = np.argmax(A_masked.reshape(P, -1), axis=1)
    max_wi, max_wj = np.unravel_index(max_flat, (W, W))

    if method == "max":
        wi, wj = max_wi.copy(), max_wj.copy()
        status = np.where(any_valid, "ok", "no_river").astype(object)
        quality = A_win  # what makes a competing cell comparable
    else:
        A_rep = plants[area_col].to_numpy(dtype=float)[:, None, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.minimum(A_win, A_rep) / np.maximum(A_win, A_rep)
        c = np.where(valid & ~np.isnan(c), c, 0.0)
        accept = valid & (c >= min_accordance)
        score = np.where(accept, (1 - c) + distance_weight * (d / d_max), np.inf)
        area_flat = np.argmin(score.reshape(P, -1), axis=1)
        area_ok = np.isfinite(score.reshape(P, -1)[np.arange(P), area_flat])
        area_wi, area_wj = np.unravel_index(area_flat, (W, W))

        wi = np.where(area_ok, area_wi, max_wi)
        wj = np.where(area_ok, area_wj, max_wj)
        status = np.where(
            ~any_valid, "no_river", np.where(area_ok, "ok", "fallback_max")
        ).astype(object)
        quality = c  # a competing branch is one that also matches the area well

    rows = np.arange(P)
    chosen_iy = cand_iy[rows, wi, wj].copy()
    chosen_ix = cand_ix[rows, wi, wj].copy()
    # plants with no valid river cell keep their nearest cell
    no_river = ~any_valid
    chosen_iy[no_river] = iy[no_river]
    chosen_ix[no_river] = ix[no_river]

    chosen_up = A[chosen_iy, chosen_ix]

    # ambiguity: a *separate* comparable candidate competes in the window. Cells
    # with quality >= ratio*chosen (uparea for "max", area accordance for "area")
    # are grouped into connected components (8-connected); a component other than
    # the chosen cell's flags the snap as ambiguous, while the chosen river's own
    # up-/downstream cells (same component) are ignored.
    chosen_q = quality[rows, wi, wj]
    big = valid & (quality >= ambiguity_ratio * chosen_q[:, None, None])
    structure = np.ones((3, 3), dtype=int)
    for p in np.flatnonzero((status == "ok") & big.any(axis=(1, 2))):
        lbl, n = label(big[p], structure=structure)
        if n > 1 and (big[p] & (lbl != lbl[wi[p], wj[p]])).any():
            status[p] = "ambiguous"

    result = plants.copy()
    result["x_snapped"] = xs[chosen_ix]
    result["y_snapped"] = ys[chosen_iy]
    result["uparea_snapped"] = chosen_up
    result["snap_distance"] = _haversine_km(lon, lat, result["x_snapped"].to_numpy(), result["y_snapped"].to_numpy())
    result["snap_status"] = status
    return result, sub


def snap_plants_to_river(
    plants,
    uparea="/tmp/atlite/glofas_uparea_v4.nc",
    method="max",
    radius=3,
    area_col="catchment_area",
    min_accordance=0.5,
    distance_weight=1.0,
    min_uparea=0.0,
    ambiguity_ratio=0.5,
    passthrough_technologies=(),
    technology_col="technology",
    rivers=None,
    river_col="river",
    river_name_col="name",
    river_aliases=None,
    river_match_dist_km=2.5,
    river_onstream_dist_km=1.5,
):
    """
    Snap plant coordinates onto GLOFAS river cells using the static uparea map.

    GLOFAS discharge is only meaningful on the cells of its river network, so
    plant coordinates that are slightly misplaced can land on a hillslope cell
    or a minor tributary. This preprocessing helper moves each plant onto a
    nearby river cell of the GLOFAS upstream-area map and appends the snapped
    coordinates (`x_snapped`, `y_snapped`) so that a subsequent
    ``cutout.hydro()`` call uses them directly.

    Parameters
    ----------
    plants : pd.DataFrame
        Run-of-river plants or dams with `lon`, `lat` columns.
    uparea : str | pathlib.Path | xr.DataArray | xr.Dataset
        GLOFAS upstream-area map. A path is opened (and downloaded if missing)
        via `atlite.datasets.glofas.retrieve_uparea`; a DataArray/Dataset is
        used directly and converted to km2 if its `units` attribute says m2.
    method : {"max", "area"}
        "max" (default) snaps to the largest-uparea cell within `radius`, i.e.
        the biggest river nearby. "area" instead snaps to the cell whose uparea
        best matches an expected catchment area given in `area_col`, which
        resolves confluences where the largest river is the wrong branch. Rows
        without a usable `area_col` value fall back to "max" behaviour.
    radius : int
        Search radius in grid cells; the window is (2*radius+1)^2 cells.
    area_col : str
        Column of `plants` holding the expected upstream catchment area in km2
        (only used by ``method="area"``).
    min_accordance : float
        For ``method="area"``, minimum ``min(A_grid, A_exp)/max(A_grid, A_exp)``
        for a cell to be accepted; below it the plant falls back to "max".
    distance_weight : float
        Weight of the normalized distance term relative to the area-mismatch
        term in the ``method="area"`` score.
    min_uparea : float
        Cells with an uparea below this value (km2) are ignored when snapping.
    ambiguity_ratio : float
        A plant is flagged as ``"ambiguous"`` if a second, non-adjacent river
        cell in the window carries at least this fraction of the chosen cell's
        uparea (a competing branch).
    passthrough_technologies : tuple/list of str
        Technologies returned UNSNAPPED (e.g. ``("Pumped Storage",)``): pumped
        storage generates from pumping, not river inflow, so a capacity-derived
        catchment area is meaningless. Matching rows get NaN snapped coordinates
        and ``snap_status="passthrough"``. Empty (default) snaps every plant.
    technology_col : str
        Column of `plants` read for `passthrough_technologies`; ignored if absent.
    rivers : geopandas.GeoDataFrame | str | pathlib.Path, optional
        Named waterway lines used for the river-aware refinement (see Notes). If
        None (default) the refinement is skipped and behaviour is unchanged.
    river_col : str
        Column of `plants` holding each plant's expected river name; ignored if
        absent (only used when `rivers` is given).
    river_name_col : str
        Name column inside `rivers`.
    river_aliases : dict, optional
        Extra ``{normalized_name: canonical}`` pairs merged onto the built-in
        cross-language alias table (see `normalize_river`).
    river_match_dist_km : float
        Maximum cell-centre-to-river-line distance (km) to label a cell with a
        river name.
    river_onstream_dist_km : float
        Tighter distance (km): cells the line runs THROUGH are preferred before
        taking the max-uparea cell, so a mislabelled bigger neighbour at a
        confluence is not chosen.

    Returns
    -------
    pd.DataFrame
        A copy of `plants` with added columns `x_snapped`, `y_snapped`
        (chosen cell centre), `uparea_snapped` (km2), `snap_distance` (km,
        great-circle), `snap_status` (one of "ok", "ambiguous", "fallback_max",
        "no_river", "passthrough", "river_match") and `matched_river` (the
        normalized river name where a river match was used, else NaN).

    Notes
    -----
    The expected catchment area for ``method="area"`` is not shipped with the
    common plant datasets and has to be estimated beforehand, e.g. from the JRC
    hydro-power database columns via the mean hydraulic discharge

        Q_mean = E_annual / (rho * g * H * eta * 8760h)

    (with head H = ``dam_height_m``, efficiency eta ~ 0.9) and dividing by a
    local specific runoff, ``A ~ Q_mean / specific_runoff``.

    River-aware refinement (``rivers`` given and `river_col` present): for each
    plant with a named river, the cells in its window are labelled with the
    nearest waterway name and the plant is moved onto the largest-uparea cell on
    its river. The on-stream distance filter is essential: near a confluence a
    ~5 km GLOFAS cell that belongs to a bigger neighbouring river can be labelled
    with the plant's river because that river's line passes nearby; restricting
    to cells the line runs through avoids grabbing the wrong big cell. The
    refinement never raises: on any failure it logs a warning and keeps the
    area/max snap.

    References
    ----------
    Godet, J., Gaume, E., Javelle, P., Nicolle, P., and Payrastre, O.:
    Technical note: Comparing three different methods for allocating river
    points to coarse-resolution hydrological modelling grid cells, Hydrol.
    Earth Syst. Sci., 28, 1403-1413, https://doi.org/10.5194/hess-28-1403-2024,
    2024.
    """
    if method not in ("max", "area"):
        raise ValueError(f'method must be "max" or "area", got "{method}".')

    cols = [
        "x_snapped",
        "y_snapped",
        "uparea_snapped",
        "snap_distance",
        "snap_status",
        "matched_river",
    ]

    # (a) pass-through: rows that must not be snapped (e.g. pumped storage)
    if len(passthrough_technologies) and technology_col in plants.columns:
        passthrough = plants[technology_col].isin(passthrough_technologies)
    else:
        passthrough = pd.Series(False, index=plants.index)
    to_snap = plants[~passthrough]

    if method == "area" and area_col not in plants.columns and len(to_snap):
        raise ValueError(f'method="area" requires the column "{area_col}" in plants.')

    result = plants.copy()
    for col in cols:
        result[col] = np.nan
    result["snap_status"] = result["snap_status"].astype(object)
    result["matched_river"] = result["matched_river"].astype(object)
    result.loc[passthrough, "snap_status"] = "passthrough"

    if len(to_snap):
        # (b) area/max snap on the non-passthrough rows
        snapped, sub = _snap_core(
            to_snap,
            uparea,
            method,
            radius,
            area_col,
            min_accordance,
            distance_weight,
            min_uparea,
            ambiguity_ratio,
        )
        snapped["matched_river"] = np.nan

        # (c) river-aware refinement onto each plant's named river
        if rivers is not None and river_col in to_snap.columns:
            snapped = _snap_river_aware(
                to_snap,
                snapped,
                sub,
                rivers,
                river_col,
                river_name_col,
                river_aliases,
                radius,
                river_match_dist_km,
                river_onstream_dist_km,
            )

        result.loc[snapped.index, cols] = snapped[cols]

    status = result["snap_status"]
    for st in ("no_river", "fallback_max", "ambiguous"):
        ids = result.index[status == st]
        if len(ids):
            logger.warning(
                f"snap_plants_to_river: {len(ids)} plant(s) with status "
                f"'{st}': {list(ids)}"
            )
    dists = result["snap_distance"].to_numpy(dtype=float)
    if np.isfinite(dists).any():
        logger.info(
            f"Snapped {int((~passthrough).sum())} plants (method={method}); "
            f"median move {np.nanmedian(dists):.1f} km, max {np.nanmax(dists):.1f} km"
        )
    return result


def _hydro_from_discharge(
    cutout,
    plants,
    time=None,
):
    """
    Get inflow time-series for `plants` from GLOFAS discharge by snapping each
    plant to a grid cell and interpolating onto the target time index.

    If `plants` carries `x_snapped`/`y_snapped` columns (from
    `snap_plants_to_river`), those cells are used directly. Otherwise each plant
    is snapped to the nearest grid cell that holds data.

    Parameters
    ----------
    plants : pd.DataFrame
        Run-of-river plants or dams with lon, lat columns. Optionally
        x_snapped, y_snapped columns pre-computed by `snap_plants_to_river`.
    time : pd.DatetimeIndex, optional
        Time index to interpolate the plant inflow onto. Defaults to the cutout's
        own time index.
    """
    if time is None:
        time = cutout.coords["time"]
    discharge = cutout.data.discharge

    if {"x_snapped", "y_snapped"}.issubset(plants.columns):
        # drop plants without a snapped cell (e.g. pumped-storage pass-through);
        # they carry no river inflow and are not returned
        snapped = plants[plants["x_snapped"].notnull() & plants["y_snapped"].notnull()]
        x = xr.DataArray(
            snapped["x_snapped"].values, dims="plant", coords={"plant": snapped.index}
        )
        y = xr.DataArray(
            snapped["y_snapped"].values, dims="plant", coords={"plant": snapped.index}
        )
        # snapped cells must sit on the cutout grid (guards against a mismatched
        # uparea grid, e.g. a v3 uparea map with a v4 cutout)
        dx = float(discharge.x[1] - discharge.x[0])
        dy = float(discharge.y[1] - discharge.y[0])
        nearest = discharge.sel(x=x, y=y, method="nearest")
        off = (np.abs(nearest.x - x) > abs(dx) / 2) | (
            np.abs(nearest.y - y) > abs(dy) / 2
        )
        if bool(off.any()):
            raise ValueError(
                "Snapped coordinates are not aligned with the cutout grid for "
                f"plants: {list(snapped.index[off.values])}"
            )
        inflow = nearest.compute()
    else:
        logger.info(
            "Snapping plants to nearest GLOFAS cell. Consider preprocessing "
            "with atlite.hydro.snap_plants_to_river() for river-aware snapping."
        )
        # snap plants to GLOFAS cells with data (cutout grid points may be all-NaN)
        present = discharge.isel(time=0).notnull()
        discharge = discharge.isel(
            x=np.flatnonzero(present.any("y").values),
            y=np.flatnonzero(present.any("x").values),
        )
        x = xr.DataArray(
            plants["lon"].values, dims="plant", coords={"plant": plants.index}
        )
        y = xr.DataArray(
            plants["lat"].values, dims="plant", coords={"plant": plants.index}
        )
        inflow = discharge.sel(x=x, y=y, method="nearest").compute()

    inflow = inflow.dropna("time", how="all").interp(time=time)
    inflow = inflow.ffill("time").bfill("time")
    return inflow.transpose("plant", "time")

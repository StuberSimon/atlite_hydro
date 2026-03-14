import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ds = xr.open_dataset("data/data_version-4.0_consolidated.nc")

# ── Selection: Central Europe, full year ──────────────────────────────────────
sel = ds.sel(
    latitude=slice(54, 48),
    longitude=slice(3, 13),
)

# ── Plot 1: xarray pcolormesh for spatial snapshot (one day) ────────────────────
snap = sel.dis24.sel(valid_time="2013-06-15")  # One snapshot
snap.plot(figsize=(10, 6), cmap="YlGnBu")
plt.plot(6.975000000006665,50.925000000002214, 'r.', markersize=10)
plt.title("River discharge central europe — 2013-06-15")
plt.savefig("scripts_ba/results/discharge_snapshot_central_europe_1.png")

# ── Plot 2: xarray line plot for time series at a single point (cologne) ────────────────
point = sel.dis24.sel(latitude=50.925000000002214, longitude=6.975000000006665, method="nearest")
point.plot(figsize=(12, 4))
plt.title("Discharge time series in Rhein (50.925°N, 6.975°E)")
plt.ylabel("dis24 (m³/s)")
plt.savefig("scripts_ba/results/discharge_rhein_2013_1.png")


cologne = point.to_dataframe().drop(columns=['surface','longitude','latitude'])

cologne_shifted = cologne.shift(1)
diff = cologne.iloc[1:-1] - cologne_shifted.iloc[1:-1]
diff_avg = diff.abs().mean().values[0]
diff_avg_rel = diff_avg / cologne.abs().mean().values[0]
diff_max = diff.abs().max().values[0]
diff_max_rel = diff_max / cologne.abs().mean().values[0]
diff_min = diff.abs().min().values[0]
diff_min_rel = diff_min / cologne.abs().mean().values[0]
print(f'- Average absolute change = {np.round(diff_avg*1, 2)} m³/s')
print(f'- Average relative change = {np.round(diff_avg_rel*100, 2)}%')
print(f'- Maximum absolute change = {np.round(diff_max*1, 2)} m³/s')
print(f'- Maximum relative change = {np.round(diff_max_rel*100, 2)}%')
print(f'- Minimum absolute change = {np.round(diff_min*1, 2)} m³/s')
print(f'- Minimum relative change = {np.round(diff_min_rel*100, 3)}%')

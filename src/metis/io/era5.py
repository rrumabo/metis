from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict, Tuple

def open_era5(path: str) -> xr.Dataset:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"[Metis] ERA5 file not found: {p}")
    ds = xr.open_dataset(p)
    return ds

def slice_time(ds: xr.Dataset, t_index: int = 0) -> xr.Dataset:
    # one timestamp: ds.isel(time=t_index) works for hourly files
    if "time" in ds.coords:
        return ds.isel(time=t_index)
    return ds

def interp_to_grid(ds: xr.Dataset, ny: int, nx: int, Ly: float, Lx: float) -> dict:
    """
    Regrid a single-time ERA5 slice to a uniform ny×nx model grid.

    Accepts either (lat, lon) or (y, x) coords in the input dataset.
    Produces fields on model coordinates y∈[0,Ly], x∈[0,Lx].
    """
    # Detect input coordinate names
    if "y" in ds.coords and "x" in ds.coords:
        y_name, x_name = "y", "x"
    else:
        # prefer 'latitude'/'longitude', fall back to 'lat'/'lon'
        y_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
        x_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
        if y_name is None or x_name is None:
            raise ValueError("interp_to_grid: dataset must have either (y,x) or (latitude/longitude) coordinates")

    # Rename to canonical (y,x) so downstream is simple
    dsn = ds.rename({y_name: "y", x_name: "x"})

    # Build target model coordinates
    y_target = np.linspace(0.0, Ly, ny)
    x_target = np.linspace(0.0, Lx, nx)

    # If input isn't already on [0,L*], map by index space (uniformize) to avoid unit headaches
    # We create normalized indices for the source grid and interpolate over (y,x) after reassigning coords
    dsn = dsn.assign_coords(
        y=np.linspace(0.0, 1.0, dsn.sizes["y"]),
        x=np.linspace(0.0, 1.0, dsn.sizes["x"]),
    )
    target = dsn.interp(
        y=np.linspace(0.0, 1.0, ny),
        x=np.linspace(0.0, 1.0, nx),
        kwargs={"fill_value": "extrapolate"},
    )

    # return plain arrays
    out = {}
    for name in ("t2m", "u10", "v10", "skt"):
        if name in target:
            out[name] = target[name].values
    return out

def make_initial_from_era5(fields: Dict[str, np.ndarray], h0: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (T_init, h_init, u_init, v_init) mapped to our solver variables:
      - T_init for heat demo (K)
      - shallow-water: h_init ~ constant + tiny bump; u_init,v_init from 10m winds (scaled)
    """
    ny, nx = next(iter(fields.values())).shape
    # Heat:
    T = fields.get("t2m", np.full((ny, nx), 300.0))  # Kelvin
    # SW:
    h  = np.full((ny, nx), h0, dtype=float)
    u  = np.zeros_like(h)
    v  = np.zeros_like(h)
    if "u10" in fields: u = 0.02 * fields["u10"]   # crude mapping m/s → m/s
    if "v10" in fields: v = 0.02 * fields["v10"]
    return T, h, u, v

def mean_bias_correction(model: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Return model - mean_bias(model, ref)."""
    bias = float(np.nanmean(model - ref))
    return model - bias
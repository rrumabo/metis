import sys
from pathlib import Path
import numpy as np
import yaml

from metis.physics.shallow_water import run_shallow_water
from metis.physics.heat import run_heat_demo          # kept import in case other modules call it
from metis.io.era5 import (
    open_era5,
    slice_time,
    interp_to_grid,
    make_initial_from_era5,
)
from metis.validation.metrics import write_metrics


def _read_yaml(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"Config file not found: {p}")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_from_config(config_path: str) -> None:
    """
    heat core: ERA5 is REQUIRED
    sw core: uses current solver; ERA5 seeding to be added in a follow-up
    """
    cfg = _read_yaml(config_path)

    grid = cfg.get("grid", {})
    runtime = cfg.get("runtime", {})
    scenario = cfg.get("scenario", {})

    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    dx = float(grid.get("dx", 100.0))
    dy = float(grid.get("dy", 100.0))

    dt = float(runtime.get("dt", 10.0))
    t_end = float(runtime.get("t_end", 3600.0))
    out_dir = Path(runtime.get("output_dir", "outputs/run")).resolve()

    core = str(cfg.get("core", "heat")).lower()
    params = cfg.get("params", {})
    era5_cfg = cfg.get("era5", {}) or {}

    print(f"[Metis] core={core} | grid={nx}x{ny} (dx={dx}, dy={dy}) | dt={dt}, t_end={t_end}")
    print(f"[Metis] scenario={scenario.get('type','baseline')}")

    _ensure_dir(out_dir)

    if core == "heat":
        # requires ERA5 for heat
        era5_path = era5_cfg.get("path")
        if not era5_path:
            raise SystemExit("[Metis] heat core requires `era5.path` in YAML (v2+).")

        ds = open_era5(era5_path)
        ds1 = slice_time(ds, int(era5_cfg.get("time_index", 0)))
        fields = interp_to_grid(ds1, ny=ny, nx=nx, Ly=ny*dy, Lx=nx*dx)
        T_init, _h0, _u0, _v0 = make_initial_from_era5(fields)

        temp = np.asarray(T_init, dtype=float)
        np.save(out_dir / "temperature.npy", temp)
        write_metrics(out_dir / "metrics.json", {
            "field": "temperature",
            "min": float(np.nanmin(temp)),
            "max": float(np.nanmax(temp)),
            "mean": float(np.nanmean(temp)),
            "source": "ERA5 regridded",
            "seeded_from_era5": bool(era5_cfg.get("path")),
        })
        print(f"[Metis] wrote ERA5-coupled temperature to {out_dir}")
        return

    # If ERA5 provided, seed initial winds from U10/V10; else fall back to internal ICs
    if era5_cfg.get("path"):
        ds = open_era5(era5_cfg["path"])
        ds1 = slice_time(ds, int(era5_cfg.get("time_index", 0)))
        fields = interp_to_grid(ds1, ny=ny, nx=nx, Ly=ny*dy, Lx=nx*dx)

        u10 = fields.get("u10"); v10 = fields.get("v10")
        if u10 is None or v10 is None:
            raise SystemExit("[Metis] sw with ERA5 requires variables u10 and v10 in the dataset")

        scale = float(params.get("wind_scale", 0.02))
        u0 = scale * np.asarray(u10, dtype=float)
        v0 = scale * np.asarray(v10, dtype=float)
        h0_val = float(params.get("h0", 1.0))
        h0 = np.full((ny, nx), h0_val, dtype=float)
        Y0 = np.stack([h0, u0, v0], axis=0)

        h, u, v = run_shallow_water(nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, t_end=t_end, params=params, Y0=Y0)
    else:
        h, u, v = run_shallow_water(nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, t_end=t_end, params=params)

    np.save(out_dir / "h.npy", h)
    np.save(out_dir / "u.npy", u)
    np.save(out_dir / "v.npy", v)
    write_metrics(str(out_dir / "metrics.json"), {
        "field": "shallow_water",
        "h_min": float(np.nanmin(h)),
        "h_max": float(np.nanmax(h)),
        "mass_sum": float(np.sum(h)),
        "core": "sw",
        "version": "v2",
        "seeded_from_era5": bool(era5_cfg.get("path")),
    })
    print(f"[Metis] wrote shallow-water fields to {out_dir}")
    return

    raise SystemExit(f"[Metis] Unknown core: {core}. Use 'heat' or 'sw'.")


def main(argv=None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "--config":
        print("Usage: python -m metis.cli.main --config path/to/config.yaml")
        raise SystemExit(1)
    print(f"[Metis] using config: {args[1]}")
    run_from_config(args[1])


if __name__ == "__main__":
    main()
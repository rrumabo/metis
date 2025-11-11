import numpy as np 
from pathlib import Path
import sys
import yaml
from metis.physics.heat import run_heat_demo
from metis.physics.shallow_water import run_sw_demo


def run_from_config(config_path: str) -> None:
    """Load a YAML config and run a dummy Metis simulation.

    v0: just writes a random temperature field to the output directory.
    v1+: this will call the real physics solvers.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    grid = cfg.get("grid", {})
    runtime = cfg.get("runtime", {})
    scenario = cfg.get("scenario", {})

    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    dt = float(runtime.get("dt", 10.0))
    t_end = float(runtime.get("t_end", 3600.0))
    out_dir = Path(runtime.get("output_dir", "outputs/demo")).resolve()

    print(f"[Metis] grid = {nx}x{ny}, dt={dt}, t_end={t_end}")
    print(f"[Metis] scenario = {scenario.get('type', 'baseline')}")

    out_dir.mkdir(parents=True, exist_ok=True)

    dx = float(grid.get("dx", 100.0))
    dy = float(grid.get("dy", 100.0))

    core = cfg.get("core","heat")
    if core == "heat":
        temp= run_heat_demo(
            nx=nx, ny=ny,
            dx=dx,
            dy=dy,
            dt=dt,
            t_end=t_end,
            base_temp=35.0,
            kappa=1e-5,
        )
        np.save(out_dir / "temperature.npy" , temp)
        print(f"[Metis] wrote temperature field to {out_dir / 'temperature.npy'}")
    
    elif core == "sw":
        params=cfg.get("params",{})
        h, u, v = run_sw_demo( 
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            dt=dt,
            t_end=t_end,
            params=params,
        )
        np.save(out_dir / "h.npy", h)
        np.save(out_dir / "u.npy", u)
        np.save(out_dir / "v.npy", v)
        print(f"[Metis] wrote shallow water fields to {out_dir}")

    else:
        print(f"[Metis] Unknown core type: {core}. Supported types are 'heat' and 'sw'--> swallow water.")
        raise SystemExit(1)
    
def main(argv=None) -> None:
    """Entry point for `python -m metis.cli.main`."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "--config":
        print("Usage: python -m metis.cli.main --config path/to/config.yaml")
        raise SystemExit(1)

    config_path = args[1]
    print(f"[Metis] using config: {config_path}")
    run_from_config(config_path)


if __name__ == "__main__":
    main()
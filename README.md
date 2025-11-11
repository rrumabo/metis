# Metis — Urban Climate Foresight Simulator

**Metis** predicts how temperature and airflow respond to adaptation measures
such as vegetation, albedo changes, or roughness modifications. It combines
fast 2D physics (shallow-water + heat) with bias-corrected data and, later,
machine-learning surrogates for rapid "what-if" runs.

## Status roadmap

- v0: repo skeleton + CLI stub
- v1: physics core (shallow-water + heat)
- v2: ERA5 loader + bias correction
- v3: scenario engine (albedo, vegetation, roughness)
- v4: validation (CFL, mass, RMSE vs data)
- v5: ROM / surrogate for fast ΔT predictions
- v6: Nicosia package (configs + report)

## Quickstart 
Clone the repo and set up your environment:

```bash
git clone https://github.com/rrumabo/metis.git
cd metis
python -m venv .venv
source .venv/bin/activate
pip install -e .

### Run a baseline heat demo:
python -m metis.cli.main --config examples/nicosia_baseline.yaml
python -m metis.validation.plots outputs/nicosia_baseline/temperature.npy

### Run a shallow-water demo:
python -m metis.cli.main --config examples/nicosia_sw.yaml
python -m metis.validation.diagnostics outputs/nicosia_sw/h.npy
python -m metis.validation.plots outputs/nicosia_sw/h.npy
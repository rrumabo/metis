# Metis — Urban Climate Foresight Simulator

A lightweight 2D physics engine for **urban climate what-if scenarios** — part of the broader  
[**Pneuma**](https://github.com/rrumabo/pneuma) digital-twin framework.

Metis focuses on **fast physical solvers** (heat and shallow-water), providing a testbed for
the data-driven and surrogate models later integrated into Pneuma. Together, they form a complete
pipeline from physical simulation → data assimilation → city-scale climate planning.

At v2, Metis serves as the ERA5-coupled physics layer within Pneuma, enabling realistic climate simulations through direct ERA5 data integration and bias correction. This milestone enhances the fidelity and applicability of urban climate forecasts.

## Status roadmap

- v0: repo skeleton + CLI stub
- v1: physics core (shallow-water + heat)
- v2: ERA5 loader + bias correction — complete, with metrics incorporating seeded_from_era5
- v3: scenario engine (albedo, vegetation, roughness)
- v4: validation (CFL, mass, RMSE vs data)
- v5: ROM / surrogate for fast ΔT predictions
- v6: Nicosia package (configs + report)

## v2 Accomplishments

- Integrated ERA5 data loader for realistic atmospheric forcing.
- Implemented bias correction methods to improve simulation accuracy.
- Developed evaluation metrics supporting seeded_from_era5 initializations.
- Established a robust foundation for data-driven extensions in Pneuma.

## Quickstart 
Clone the repo and set up your environment:

```bash
git clone https://github.com/rrumabo/metis.git
cd metis
python -m venv .venv
source .venv/bin/activate
pip install -e .

### Run a v2 heat demo:
python -m metis.cli.main --config examples/nicosia_v2_heat.yaml
python -m metis.validation.plots outputs/nicosia_v2_heat/temperature.npy

### Run a v2 shallow-water demo:
python -m metis.cli.main --config examples/nicosia_v2_sw.yaml
python -m metis.validation.diagnostics outputs/nicosia_v2_sw/h.npy
python -m metis.validation.plots outputs/nicosia_v2_sw/h.npy
```
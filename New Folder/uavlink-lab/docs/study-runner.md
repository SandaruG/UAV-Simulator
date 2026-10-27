# Running systematic UAV radio studies

`exhaustive_experiment.py` is the top-level experiment runner. It enumerates technology combinations, runs each policy and seed, resumes from its append-only ledger, and writes CSV and structured JSON summaries.

## Commands

```bash
# Show the native drone_multilink catalog and legacy aliases
python3 exhaustive_experiment.py --list-techs

# Inspect combinations, filtered combinations, and the runtime estimate
python3 exhaustive_experiment.py --config study_demo.json --dry-run --out results/demo

# Fast mode uses uavlink_sim.LinkCfg models; it is the default
python3 exhaustive_experiment.py --config study_demo.json --backend fast --out results/demo

# High-fidelity mode runs the flight, environment, radio, antenna, codec, and traffic models
python3 exhaustive_experiment.py --config study_demo.json --backend high_fidelity --out results/demo-hifi

# Continue an interrupted run with its original study definition
python3 exhaustive_experiment.py --config study_demo.json --backend high_fidelity --out results/demo-hifi --resume
```

Both `python3 exhaustive_experiment.py list-techs` and `python3 exhaustive_experiment.py run --config ...` remain supported. The shorter `--list-techs` and implicit `run` forms are also accepted.

## Technology names

Study IDs are the canonical names in `drone_multilink.radio_catalog()`. The legacy IDs are accepted as aliases and normalize to these native names before combination generation:

| Legacy study / link name | Canonical radio |
|---|---|
| `analog_fpv_58` | `fpv58_analog` |
| `digital_58_dji` | `ocusync` |
| `digital_58_openhd`, `wifi_5g` | `wifi` |
| `ism_24_ctrl` | `elrs24` |
| `ism_24_wifi` | `rf24` |
| `lr_12` | `lr12g` |
| `sub_868_900`, `sub1ghz` | `rf900` |
| `band_17`, `band_33`, `band_72` | `lr17g`, `lr33g`, `lr72g` |
| `cell_lte`, `lte` | `cell` |
| `cell_5g` | `nr_a` |
| `cell_5g_multisim` | `sim_bond` |
| `sat_leo_starlink`, `sat_leo` | `leo` |
| `sat_kuka_leo`, `sat_oneweb_ku` | `oneweb_ku` |
| `sat_iridium` | `iridium_certus` |
| `sat_sband_leo` | `globalstar_s` |
| `sat_lband_geo` | `inmarsat_l` |
| `sat_kuka_geo` | `geo_ku` |
| `mesh_rf` | `mesh_manet` |

The mapping is bidirectional: the alias table is `uavlink_integration.RADIO_ALIASES`, and `reverse_radio_aliases()` groups accepted older names by canonical radio. Aliases are resolved for technology selection, exclusions, requirements, policy priorities, and legacy fault-link names. Native-only radios such as `walksnail`, `hdzero`, `lte_a`, `lte_b`, `nr_b`, `geo_ka`, and `sdr_ofdm` stay individually selectable.

`techs.overrides` overrides fast-mode `uavlink_sim.LinkCfg` values. `techs.radio_overrides` independently accepts `drone_multilink.RadioParams` field overrides and applies in high-fidelity mode. Use `--list-techs` to see labels, roles, nominal frequency, bandwidth, power, and latency.

## Backends and policies

- `execution.backend: "fast"` (default) uses the existing scenario/controller and fast per-link abstractions in `uavlink_sim`. It supports all technologies through `LinkCfg` approximations and the existing evaluator.
- `execution.backend: "high_fidelity"` (or `--backend high_fidelity`) creates and enables the selected native radio objects and runs `drone_multilink.DroneMultiLinkSim`. `execution.physics_dt_s` controls flight integration; `execution.radio_dt_s` controls radio updates (defaults to `scale.dt`, with a 20 ms minimum).
- High-fidelity routing uses Part 3 `predictive_handover` policies. `reactive`, `bonding`, `static_primary`, `pred_threshold`, and `pred_cost` are supported, as are custom `predictive_handover.DecisionPolicy` classes. The controller can route control, duplicate control, video, and bonded video across enabled radios. `video_modes: ["ai"]` selects the native neural codec.
- In high-fidelity mode the native simulator's summary supplies directly measured metrics. Metrics not emitted by that model are left missing rather than estimated from unrelated fast-model values. The run row identifies `backend`, active technologies, native radios, and unsupported scenario fault types.
- High-fidelity `comm_power_mean_w` and `energy_comm_wh` are integrated at each radio update from the combined native radio and encoder draw; `radio_energy_wh` in the structured run summary remains available as the per-radio hardware-only breakdown.

The built-in scenario geometry, waypoints, weather, buildings, stations, interference, and random trial perturbations are adapted into the native model. Not all `uavlink_sim` fault injection mechanisms have equivalents in `drone_multilink`; check `unsupported_faults` in each result row before interpreting a study. The simulation outputs are model results, not hardware calibration.

## Outputs and precision

Each output directory contains `results.csv` (one row per valid combination, policy, video mode, and seed), `runs.jsonl` (append-only, full Python float precision and resume ledger), `results.json`, `summary.json`, `ranking_blind.csv`, `ranking_efficiency.csv`, `pareto.csv`, and `summary.md`. `metrics.decimals` controls CSV display precision (default six decimals); choose `"repr"` for round-trip float strings. The ranked summaries compare blind time and quality per watt. With `metrics.save_ticks` set to `first:N` or `all`, per-run time series are stored under `logs/`.

Use `--dry-run` to inspect plan counts, rejected combinations, and an optional short runtime calibration before any experiment runs. Resume only with the same config and output directory; a changed plan hash is rejected by default.

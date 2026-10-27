#!/usr/bin/env python3
"""Run one or more UAV handover experiments and evaluate the resulting batch.

The run path composes uavlink_sim's scenario/simulation and logging APIs with
uavlink_eval's report pipeline. The ``predictive_handover`` policy name selects
the adapter registered by uavlink_sim, which delegates decisions to Part 3.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import drone_multilink
import predictive_handover
import uavlink_eval
import uavlink_sim


LOG = logging.getLogger("uavlink.experiment")
_PREDICTIVE_ALIASES = {"cost", "predictive", "predictive_handover", "handover_cost"}


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON config {path!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON config {path!r} must contain an object")
    return value


def _normalize_policy(spec: Any) -> Any:
    """Map convenient names to the native Part 3 adapter, preserving other specs."""
    if isinstance(spec, str):
        if spec in _PREDICTIVE_ALIASES:
            return {"name": "predictive_handover", "params": {"policy_name": "cost"}}
        return spec
    if not isinstance(spec, dict):
        raise ValueError(f"invalid policy specification {spec!r}")
    normalized = copy.deepcopy(spec)
    name = normalized.get("name")
    if name in _PREDICTIVE_ALIASES:
        params = normalized.setdefault("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"policy {name!r} params must be an object")
        params.setdefault("policy_name", "cost")
        normalized["name"] = "predictive_handover"
    return normalized


def _normalize_batch_config(source: dict, args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Accept the existing uavlink_sim batch schema plus runner-specific sections."""
    if "batch" in source:
        batch = copy.deepcopy(source["batch"])
    elif "uavlink_sim" in source:
        batch = copy.deepcopy(source["uavlink_sim"])
    else:
        batch = {k: copy.deepcopy(v) for k, v in source.items()
                 if k not in ("evaluation", "uavlink_eval", "predictive_handover")}
    if not isinstance(batch, dict):
        raise ValueError("batch/uavlink_sim config must be a JSON object")
    eval_raw = source.get("evaluation", source.get("uavlink_eval", {}))
    predictive_cfg = source.get("predictive_handover", {})
    if not isinstance(eval_raw, dict) or not isinstance(predictive_cfg, dict):
        raise ValueError("evaluation and predictive_handover sections must be JSON objects")

    if args.scenario is not None:
        batch["scenario"] = args.scenario
    if args.policy is not None:
        batch["policies"] = [{"label": args.policy, "name": args.policy}]
    if args.seeds is not None:
        if args.seeds < 1:
            raise ValueError("--seeds must be at least 1")
        batch["n_trials"] = args.seeds
    if args.seed is not None:
        batch["master_seed"] = args.seed
    if args.duration is not None:
        batch.setdefault("overrides", {})["duration_s"] = args.duration

    if "scenario" not in batch:
        batch["scenario"] = "urban_canyon"
    if "policies" not in batch:
        batch["policies"] = [{"label": "predictive", "name": "predictive_handover"}]
    if "n_trials" not in batch:
        batch["n_trials"] = 1
    if "master_seed" not in batch:
        batch["master_seed"] = 0
    if "save_ticks" not in batch:
        batch["save_ticks"] = "all" if batch["n_trials"] == 1 else "first:2"

    if not isinstance(batch["policies"], list) or not batch["policies"]:
        raise ValueError("policies must be a non-empty list")
    for policy in batch["policies"]:
        if isinstance(policy, dict) and policy.get("name") in (_PREDICTIVE_ALIASES | {"predictive_handover"}):
            params = policy.setdefault("params", {})
            if not isinstance(params, dict):
                raise ValueError("predictive policy params must be an object")
            params.setdefault("handover_config", predictive_cfg.get("config", {}))
            params.setdefault("link_specs", predictive_cfg.get("link_specs", {}))
            if "policy" in predictive_cfg:
                params.setdefault("policy_name", predictive_cfg["policy"])
    batch["policies"] = [_normalize_policy(policy) for policy in batch["policies"]]
    batch.setdefault("name", "uavlink_experiment")
    batch.setdefault("baseline", None)
    return batch, eval_raw, predictive_cfg


def _validate(batch: dict, eval_raw: dict, args: argparse.Namespace) -> tuple[Any, Any]:
    if int(batch.get("n_trials", 0)) < 1:
        raise ValueError("n_trials must be at least 1")
    if int(batch.get("master_seed", 0)) < 0:
        raise ValueError("master_seed must be non-negative")
    scenario = uavlink_sim.load_scenario(batch["scenario"])
    if "overrides" in batch:
        raw = uavlink_sim.load_scenario_dict(batch["scenario"])
        raw = uavlink_sim.deep_merge(raw, batch["overrides"])
        scenario = uavlink_sim.load_scenario(raw)

    for spec in batch["policies"]:
        policy = uavlink_sim.make_policy(spec)
        if policy.name == "predictive_handover":
            cfg = policy.handover_config
            predictive_handover.HandoverConfig.from_dict(cfg)
    eval_cfg = uavlink_eval.from_dict(uavlink_eval.EvalConfig, eval_raw)
    if args.no_plots:
        eval_cfg.plots.enabled = False
    return scenario, eval_cfg


def _dry_run(batch: dict, eval_cfg: Any, scenario: Any) -> None:
    jobs = len(uavlink_sim.expand_policies(batch["policies"])) * int(batch["n_trials"])
    LOG.info("Configuration is valid")
    LOG.info("Scenario %s: %.3f s duration, dt=%.3f s", scenario.name, scenario.duration_s, scenario.dt)
    LOG.info("%d trial(s), %d policy/policy-sweep job(s)", batch["n_trials"], jobs)
    LOG.info("Evaluation plots: %s", "enabled" if eval_cfg.plots.enabled else "disabled")
    for policy in batch["policies"]:
        LOG.info("Policy: %s", policy.get("name", policy) if isinstance(policy, dict) else policy)


def _self_test() -> bool:
    """Fast cross-module smoke test; avoids the long statistical benchmark suites."""
    checks: list[tuple[str, bool]] = []
    try:
        env = __import__("radioenv", fromlist=["RadioEnvironment", "presets"])
        radio = env.RadioEnvironment(env.presets.get("demo", seed=2))
        q = radio.query_all((0.0, 0.0, 100.0), 0.0, fading="none")
        checks.append(("radioenv map generation and link query", bool(q)))
    except Exception:
        LOG.exception("radioenv smoke test failed")
        checks.append(("radioenv map generation and link query", False))

    try:
        cfg = drone_multilink.SimConfig(physics_dt=0.01, radio_dt=0.02, log_hz=5.0)
        model = drone_multilink.DroneMultiLinkSim(cfg)
        model.run(0.05)
        checks.append(("drone_multilink initialization and stepping", model.t > 0))
    except Exception:
        LOG.exception("drone_multilink smoke test failed")
        checks.append(("drone_multilink initialization and stepping", False))

    try:
        scenario = uavlink_sim.load_scenario_dict("urban_canyon")
        scenario["duration_s"], scenario["dt"] = 0.5, 0.05
        scenario.setdefault("traffic", {})["ctrl_hz"] = 20.0
        sim = uavlink_sim.SimulationController(
            scenario,
            {"name": "predictive_handover", "params": {"policy_name": "cost"}},
            seed=11,
        ).run()
        result = sim.result()
        checks.append(("predictive_handover -> uavlink_sim policy contract", sim.done and result.outcome != "running"))
    except Exception:
        LOG.exception("predictive policy integration test failed")
        checks.append(("predictive_handover -> uavlink_sim policy contract", False))

    try:
        ticks = uavlink_eval._synthetic_ticks(dt=0.1, T=1.0)
        result = uavlink_eval.evaluate_run(ticks, [], [], {}, uavlink_eval.EvalConfig(), {})
        checks.append(("uavlink_eval accepts simulator-shaped ticks", "scalars" in result))
    except Exception:
        LOG.exception("uavlink_eval smoke test failed")
        checks.append(("uavlink_eval accepts simulator-shaped ticks", False))

    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(passed for _, passed in checks)


def _run(batch: dict, eval_cfg: Any, out: str, args: argparse.Namespace) -> int:
    out_path = Path(out).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    LOG.info("Running %d job(s) into %s", len(uavlink_sim.make_jobs(batch)), out_path)
    summary = uavlink_sim.run_batch(
        batch,
        str(out_path),
        workers=args.workers,
        resume=args.resume,
        quiet=args.quiet,
    )
    evaluation_out = out_path / "evaluation"
    report = uavlink_eval.evaluate(
        batch=str(out_path),
        cfg=eval_cfg,
        out=str(evaluation_out),
        workers=args.workers,
        quiet=args.quiet,
    )
    policy_count = len(summary.get("policies", {}))
    failed_runs = int(summary.get("n_errors", 0))
    failed_evals = int(report["info"].get("n_errors", 0))
    print(f"\nCompleted {batch['n_trials']} trial(s) across {policy_count} policy/policy-sweep label(s).")
    print(f"Simulation errors: {failed_runs}; evaluation errors: {failed_evals}.")
    print(f"Batch summary: {out_path / 'summary.md'}")
    print(f"Evaluation report: {evaluation_out / 'report.md'}")
    print(f"Evaluation tables, metrics, and figures: {evaluation_out}")
    if report.get("figures"):
        print(f"Generated {len(report['figures'])} figure(s).")
    return 1 if failed_runs or failed_evals else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch", action="store_true", help="read a Monte-Carlo batch JSON config")
    mode.add_argument("--self-test", action="store_true", help="run short cross-module integration smoke tests")
    parser.add_argument("--config", help="batch/config JSON file")
    parser.add_argument("--dry-run", action="store_true", help="validate config and report planned work only")
    parser.add_argument("--scenario", help="simulator preset name or scenario JSON file")
    parser.add_argument("--policy", help="single policy name (default: predictive_handover/cost)")
    parser.add_argument("--seeds", type=int, help="number of trials for the selected policy")
    parser.add_argument("--seed", type=int, help="master seed (default: 0)")
    parser.add_argument("--duration", type=float, help="override simulation duration in seconds")
    parser.add_argument("--out", default="results/experiment", help="output directory")
    parser.add_argument("--workers", type=int, default=1, help="worker processes for simulation and evaluation")
    parser.add_argument("--resume", action="store_true", help="resume an existing batch")
    parser.add_argument("--no-plots", action="store_true", help="write tables/reports but skip figures")
    parser.add_argument("--quiet", action="store_true", help="reduce progress output")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.self_test:
        return 0 if _self_test() else 1
    if args.batch and not args.config:
        parser.error("--batch requires --config")
    if args.config:
        source = _load_json(args.config)
    else:
        source = {}
    try:
        batch, eval_raw, _ = _normalize_batch_config(source, args)
        scenario, eval_cfg = _validate(batch, eval_raw, args)
        if args.dry_run:
            _dry_run(batch, eval_cfg, scenario)
            return 0
        return _run(batch, eval_cfg, args.out, args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        LOG.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOG.error("Experiment interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())

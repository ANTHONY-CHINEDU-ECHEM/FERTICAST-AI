"""Command line entry point: ``python -m ferticast.cli <command>``."""
from __future__ import annotations

import argparse
import json
import logging

from ferticast.config import load_config, resolve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ferticast", description="FertiCast AI pipeline")
    parser.add_argument("command", choices=["simulate", "train", "report", "all", "predict"])
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file")
    parser.add_argument("--record", default=None, help="JSON patient record for the predict command")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)

    if args.command in {"simulate", "all"}:
        from ferticast.data.simulate import simulate_register

        path = resolve(cfg.data.raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = simulate_register(cfg.data.n_cycles, cfg.data.n_clinics, cfg.data.year_min, cfg.data.year_max, cfg.seed)
        frame.to_csv(path, index=False)
        logging.info("Wrote %d rows to %s", len(frame), path)
    if args.command in {"train", "all"}:
        from ferticast.models.train import run_training

        report = run_training(cfg)
        print(json.dumps(report["metrics"]["FertiCast final"], indent=2))
    if args.command in {"report", "all"}:
        from ferticast.models.report import generate_figures

        print("Figures:", ", ".join(generate_figures(cfg)))
    if args.command == "predict":
        from ferticast.inference.predictor import LiveBirthPredictor

        record = json.loads(args.record) if args.record else {"female_age": 34, "infertility_cause": "unexplained"}
        print(json.dumps(LiveBirthPredictor.load().explain(record), indent=2))


if __name__ == "__main__":
    main()

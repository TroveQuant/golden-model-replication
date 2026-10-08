"""Unified command-line entrypoint for the Golden Model adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from golden_model.config import load_config  # noqa: E402
from golden_model.rqalpha_provider import RQAlphaProvider  # noqa: E402
from golden_model.pipeline import generate_artifacts  # noqa: E402
from golden_model.report import validate_report  # noqa: E402
from golden_model.validation import validate_project_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the adapted public-data replication of the Golden Model."
    )
    parser.add_argument(
        "command",
        choices=("validate", "fetch", "backtest", "report", "validate-report", "all"),
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Repository-relative YAML configuration path.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore provider cache and retrieve source observations again.",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)

    if args.command == "validate":
        summary = validate_project_config(config, ROOT)
        print("Configuration valid.")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0

    if args.command == "fetch":
        validate_project_config(config, ROOT)
        bundle = RQAlphaProvider(config, ROOT).load(refresh=args.refresh)
        print("Real RQAlpha bundle and official PIT membership data ready.")
        for key, value in bundle.provenance.items():
            if key in {"bundle", "historical_membership", "cache_files"}:
                continue
            print(f"{key}: {value}")
        return 0

    if args.command in {"backtest", "report", "all"}:
        validate_project_config(config, ROOT)
        result, validation = generate_artifacts(config, ROOT, refresh=args.refresh)
        print("Backtest artifacts generated from real provider data.")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-report":
        output_dir = ROOT / config["outputs"]["directory"]
        validation = validate_report(output_dir, config)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    raise SystemExit(
        f"Command '{args.command}' is not available until its implementation stage is complete."
    )


if __name__ == "__main__":
    raise SystemExit(main())

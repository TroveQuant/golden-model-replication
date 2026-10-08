"""End-to-end artifact generation for the adapted replication."""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
import json
import os
import platform
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd
import yaml

from .backtest import BacktestResult, run_backtest
from .data_provider import validate_market_data, write_json
from .migration_validation import run_migration_validation
from .portfolio import build_target_weights
from .report import build_report_html, validate_report
from .rqalpha_provider import RQAlphaProvider
from .sentiment import calculate_sentiment
from .signals import generate_signals


def _git_sha(root: Path) -> str:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-initialized"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def generate_artifacts(
    config: dict[str, Any], root: Path, refresh: bool = False
) -> tuple[BacktestResult, dict[str, Any]]:
    provider = RQAlphaProvider(config, root)
    bundle = provider.load(refresh=refresh)
    effective_config = deepcopy(config)
    effective_config["data"]["effective_end_date"] = bundle.provenance["actual_end"]
    validation = validate_market_data(bundle, effective_config)
    migration = run_migration_validation(bundle, effective_config, root)
    sentiment = calculate_sentiment(bundle.stock_prices, bundle.constituents, effective_config)
    signals = generate_signals(sentiment, effective_config)
    targets = build_target_weights(signals, effective_config)
    result = run_backtest(bundle.index_prices, targets, effective_config)

    output_dir = (root / config["outputs"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = config["outputs"]
    _write_csv(result.equity_curve, output_dir / outputs["equity_curve"])
    _write_csv(result.positions, output_dir / outputs["positions"])
    _write_csv(result.trades, output_dir / outputs["trades"])
    _write_csv(sentiment, output_dir / outputs["sentiment"])
    _write_csv(signals, output_dir / outputs["signals"])
    write_json(output_dir / outputs["metrics"], result.metrics)
    write_json(output_dir / outputs["migration_validation"], migration)

    provenance = dict(bundle.provenance)
    provenance["validation"] = validation
    provenance["migration_validation"] = migration
    provenance["run"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_sha(root),
        "command": "python run.py all" + (" --refresh" if refresh else ""),
        "production_data": True,
        "synthetic_fallback": False,
        "runtime": f"Python {platform.python_version()} on {platform.platform()}",
    }
    write_json(output_dir / outputs["provenance"], provenance)

    resolved = {
        key: value for key, value in effective_config.items() if not key.startswith("_")
    }
    (output_dir / outputs["resolved_config"]).write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    payload = {
        "project": config["project"],
        "headline_model": config["strategy"]["headline_model"],
        "metrics": result.metrics,
        "data_validation": validation,
        "migration_validation": migration,
        "commit_sha": provenance["run"]["commit_sha"],
    }
    write_json(output_dir / outputs["report_payload"], payload)
    report_html = build_report_html(
        effective_config,
        result.equity_curve,
        result.positions,
        result.trades,
        result.metrics,
        provenance,
    )
    (output_dir / outputs["report_html"]).write_text(report_html, encoding="utf-8")
    report_validation = validate_report(output_dir, effective_config)
    return result, {
        "data": validation,
        "migration": {"status": migration["status"], "checks": migration["checks"]},
        "report": report_validation,
    }

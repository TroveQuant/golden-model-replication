"""Configuration and production-safety validation."""

from __future__ import annotations

from datetime import date
import importlib.util
from pathlib import Path
from typing import Any


FORBIDDEN_PRODUCTION_PROVIDERS = {
    "demo",
    "fixture",
    "mock",
    "random",
    "sample",
    "synthetic",
}


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid configuration section: {key}")
    return value


def _parse_date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD).") from exc


def validate_project_config(config: dict[str, Any], root: Path) -> dict[str, Any]:
    project = _require_mapping(config, "project")
    data = _require_mapping(config, "data")
    strategy = _require_mapping(config, "strategy")
    backtest = _require_mapping(config, "backtest")
    outputs = _require_mapping(config, "outputs")
    automation = _require_mapping(config, "automation")
    migration = _require_mapping(config, "migration_validation")

    if project.get("status") != "adapted":
        raise ValueError("Project status must remain 'adapted'.")
    provider = str(data.get("provider", "")).strip().lower()
    if not provider or provider in FORBIDDEN_PRODUCTION_PROVIDERS:
        raise ValueError(f"Forbidden or missing production data provider: {provider!r}")
    if provider != "rqalpha":
        raise ValueError("The confirmed production provider is the RQAlpha free bundle.")
    if int(data.get("expected_historical_securities", 0)) != 1458:
        raise ValueError("The audited official history must contain exactly 1458 securities.")
    workers = int(data.get("max_parallel_workers", 1))
    if workers < 1 or workers > 8:
        raise ValueError("data.max_parallel_workers must be between 1 and 8.")
    timeout_seconds = int(data.get("network_timeout_seconds", 0))
    if timeout_seconds < 5 or timeout_seconds > 120:
        raise ValueError("data.network_timeout_seconds must be between 5 and 120.")

    start = _parse_date(data.get("start_date"), "data.start_date")
    end = _parse_date(data.get("end_date"), "data.end_date")
    if start >= end:
        raise ValueError("data.start_date must be before data.end_date.")
    for universe, definition in data.get("universes", {}).items():
        expected = int(definition.get("expected_constituents", 0))
        required_count = 50 if universe == "sse50" else 500 if universe == "csi500" else 0
        if expected != required_count:
            raise ValueError(
                f"{universe} must fail closed at exactly {required_count} constituents."
            )
        official_code = str(definition.get("official_index_code", ""))
        if len(official_code) != 6 or not official_code.isdigit():
            raise ValueError(f"Missing official index code for {universe}.")
        verified_dates = definition.get("verified_change_dates", [])
        parsed_dates = [
            _parse_date(value, f"data.universes.{universe}.verified_change_dates")
            for value in verified_dates
        ]
        if parsed_dates != sorted(set(parsed_dates)):
            raise ValueError(
                f"Verified constituent change dates for {universe} must be unique and sorted."
            )
        if any(value < start or value > end for value in parsed_dates):
            raise ValueError(
                f"Verified constituent change dates for {universe} must be inside the sample."
            )
    if strategy.get("rolling_window") != 3 or strategy.get("state_window") != 5:
        raise ValueError("Confirmed strategy windows are 3 and 5 trading days.")
    if backtest.get("signal_at") != "close" or backtest.get("execute_at") != "next_close":
        raise ValueError("Confirmed timing is close signal with next-close execution.")
    if backtest.get("execution_delay_sessions") != 1:
        raise ValueError("Execution delay must be exactly one trading session.")
    if float(backtest.get("one_way_cost_bps", -1)) < 0:
        raise ValueError("Transaction cost cannot be negative.")
    if float(backtest.get("one_way_cost_bps", -1)) != 5.0:
        raise ValueError("Confirmed one-way transaction cost is exactly 5 bps.")
    weights = strategy.get("weights", {})
    if (
        float(weights.get("sse50_long", 0)) != 0.40
        or float(weights.get("csi500_short", 0)) != -0.40
        or float(weights.get("cash", 0)) != 0.20
    ):
        raise ValueError("Confirmed allocation is +40%/-40% with a 20% cash sleeve.")

    output_dir = root / str(outputs.get("directory", ""))
    cache_dir = root / str(data.get("cache_dir", ""))
    try:
        output_dir.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Output directory must remain inside the repository.") from exc
    try:
        cache_dir.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Cache directory must remain inside the repository.") from exc
    certified_audit = root / str(migration.get("certified_audit_path", ""))
    try:
        certified_audit.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Certified migration audit must remain inside the repository.") from exc
    if not str(migration.get("certified_audit_path", "")).strip():
        raise ValueError("A certified migration audit path is required for clean CI.")
    if migration.get("allow_certified_audit_without_reference") is not True:
        raise ValueError("Clean CI must explicitly use the certified migration audit gate.")

    secrets = automation.get("required_secrets")
    if secrets not in ([], None):
        if not isinstance(secrets, list) or not all(isinstance(item, str) for item in secrets):
            raise ValueError("automation.required_secrets must be a list of environment names.")

    required_modules = ("h5py", "numpy", "pandas", "pypdf", "requests", "rqalpha", "xlrd")
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise ValueError(f"Missing runtime dependencies: {missing_modules}")

    return {
        "project": project["name"],
        "mode": project["replication_mode"],
        "provider": provider,
        "sample": f"{start.isoformat()} to {end.isoformat()}",
        "timing": "T close signal -> T+1 close execution",
        "membership": "official event-sourced SSE/CSI notices, fail closed",
        "baostock": "migration validation only",
        "config": config["_config_path"],
    }

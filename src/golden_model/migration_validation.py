"""Read-only migration comparison between RQAlpha and existing BaoStock cache."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_provider import MarketDataBundle, sha256_file, write_json
from .sentiment import calculate_sentiment
from .signals import generate_signals


def _load_certified_audit(
    path: Path,
    settings: dict[str, Any],
    actual_end: str,
) -> dict[str, Any]:
    """Validate the committed migration gate used by cache-free CI runners."""

    if not path.is_file():
        raise ValueError(f"Certified migration audit not found: {path}.")
    result = json.loads(path.read_text(encoding="utf-8"))
    required_checks = {
        "sign_close_minus_open",
        "sign_close_minus_preclose",
        "suspension",
        "relative_50_500_signal",
    }
    checks = result.get("checks", {})
    if result.get("status") != "passed" or not all(
        checks.get(key) is True for key in required_checks
    ):
        raise ValueError("Certified migration audit is not a complete passing audit.")
    if result.get("production_provider") != "RQAlpha free monthly bundle":
        raise ValueError("Certified migration audit has the wrong production provider.")
    if int(result.get("reference_security_files", 0)) < int(
        settings["minimum_reference_securities"]
    ):
        raise ValueError("Certified migration audit has insufficient reference securities.")
    if pd.Timestamp(result.get("matched_end_date")) < pd.Timestamp(actual_end):
        raise ValueError("Certified migration audit does not cover the production end date.")
    output = dict(result)
    output["gate_mode"] = "certified_read_only_audit"
    output["certified_audit_sha256"] = sha256_file(path)
    return output


def _reference_directory(root: Path, minimum: int) -> tuple[Path, list[Path]]:
    candidates: list[tuple[int, Path, list[Path]]] = []
    if root.is_dir():
        for directory in root.iterdir():
            if directory.is_dir():
                files = sorted(directory.glob("*.csv.gz"))
                candidates.append((len(files), directory, files))
    if not candidates:
        raise ValueError(f"No existing BaoStock migration cache under {root}.")
    count, directory, files = max(candidates, key=lambda item: item[0])
    if count < minimum:
        raise ValueError(
            f"BaoStock migration cache has {count} securities, below required {minimum}."
        )
    return directory, files


def _load_reference(files: list[Path]) -> pd.DataFrame:
    required = {"date", "code", "open", "close", "preclose", "tradestatus"}
    pieces: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if missing := required.difference(frame.columns):
            raise ValueError(f"BaoStock cache {path.name} missing {sorted(missing)}.")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        pieces.append(frame)
    output = pd.concat(pieces, ignore_index=True)
    if output.duplicated(["date", "code"]).any():
        raise ValueError("BaoStock migration cache contains duplicate date/code rows.")
    return output


def _agreement(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    return float((left[mask] == right[mask]).mean()) if mask.any() else float("nan")


def _sign_agreement(left: pd.Series, right: pd.Series) -> float:
    return _agreement(np.sign(pd.to_numeric(left, errors="coerce")), np.sign(pd.to_numeric(right, errors="coerce")))


def _price_stats(merged: pd.DataFrame, field: str, tolerance: float) -> dict[str, Any]:
    left = pd.to_numeric(merged[f"{field}_rq"], errors="coerce")
    right = pd.to_numeric(merged[f"{field}_bs"], errors="coerce")
    mask = left.notna() & right.notna() & right.ne(0)
    relative = ((left[mask] - right[mask]).abs() / right[mask].abs())
    return {
        "observations": int(mask.sum()),
        "median_absolute_relative_difference": float(relative.median()),
        "p99_absolute_relative_difference": float(relative.quantile(0.99)),
        "within_relative_tolerance_rate": float(relative.le(tolerance).mean()),
        "tolerance": tolerance,
    }


def run_migration_validation(
    bundle: MarketDataBundle,
    config: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Compare providers without ever adding BaoStock rows to production data."""

    settings = config["migration_validation"]
    reference_root = root / settings["baostock_cache_root"]
    has_reference_files = reference_root.is_dir() and any(
        reference_root.glob("*/*.csv.gz")
    )
    if not has_reference_files:
        if not bool(settings.get("allow_certified_audit_without_reference", False)):
            raise ValueError(f"No existing BaoStock migration cache under {reference_root}.")
        return _load_certified_audit(
            root / settings["certified_audit_path"],
            settings,
            str(bundle.provenance["actual_end"]),
        )
    directory, files = _reference_directory(
        reference_root, int(settings["minimum_reference_securities"])
    )
    baostock = _load_reference(files)
    rq = bundle.stock_prices.copy()
    shared_codes = set(baostock["code"].astype(str)) & set(rq["code"].astype(str))
    if len(shared_codes) < int(settings["minimum_reference_securities"]):
        raise ValueError(
            f"Only {len(shared_codes)} BaoStock securities overlap RQAlpha production data."
        )
    baostock = baostock.loc[baostock["code"].isin(shared_codes)].copy()
    rq = rq.loc[rq["code"].isin(shared_codes)].copy()
    merged = rq.merge(
        baostock,
        on=["date", "code"],
        how="inner",
        suffixes=("_rq", "_bs"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Migration comparison has no overlapping date/code observations.")

    tolerance = float(settings["price_relative_tolerance"])
    open_stats = _price_stats(merged, "open", tolerance)
    close_stats = _price_stats(merged, "close", tolerance)
    preclose_stats = _price_stats(merged, "preclose", tolerance)
    open_close_sign = _sign_agreement(
        merged["close_rq"] - merged["open_rq"],
        merged["close_bs"] - merged["open_bs"],
    )
    close_preclose_sign = _sign_agreement(
        merged["close_rq"] - merged["preclose_rq"],
        merged["close_bs"] - merged["preclose_bs"],
    )
    suspension_agreement = _agreement(
        pd.to_numeric(merged["tradestatus_rq"], errors="coerce"),
        pd.to_numeric(merged["tradestatus_bs"], errors="coerce"),
    )

    # Sentiment is evaluated on the same shared securities and the same official
    # PIT membership intervals, preventing coverage differences from masquerading
    # as a provider discrepancy.
    shared_membership = bundle.constituents.loc[
        bundle.constituents["code"].isin(shared_codes)
    ].copy()
    rq_compare = rq[["date", "code", "open", "close", "preclose", "tradestatus"]].copy()
    bs_compare = baostock[["date", "code", "open", "close", "preclose", "tradestatus"]].copy()
    rq_sentiment = calculate_sentiment(rq_compare, shared_membership, config)
    bs_sentiment = calculate_sentiment(bs_compare, shared_membership, config)
    emotion = rq_sentiment.merge(
        bs_sentiment,
        on=["date", "model", "universe"],
        suffixes=("_rq", "_bs"),
    )
    emotion_sign_agreement = _sign_agreement(
        emotion["smoothed_score_rq"], emotion["smoothed_score_bs"]
    )
    emotion_correlation = float(
        emotion[["smoothed_score_rq", "smoothed_score_bs"]].dropna().corr().iloc[0, 1]
    )
    rq_signals = generate_signals(rq_sentiment, config)
    bs_signals = generate_signals(bs_sentiment, config)
    signals = rq_signals.merge(
        bs_signals, on=["date", "model"], suffixes=("_rq", "_bs")
    )
    signal_agreement = _agreement(signals["direction_rq"], signals["direction_bs"])

    checks = {
        "sign_close_minus_open": open_close_sign
        >= float(settings["minimum_open_close_sign_agreement"]),
        "sign_close_minus_preclose": close_preclose_sign
        >= float(settings["minimum_close_preclose_sign_agreement"]),
        "suspension": suspension_agreement
        >= float(settings["minimum_suspension_agreement"]),
        "relative_50_500_signal": signal_agreement
        >= float(settings["minimum_signal_agreement"]),
    }
    result: dict[str, Any] = {
        "status": "passed" if all(checks.values()) else "failed",
        "gate_mode": "direct_read_only_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_provider": "RQAlpha free monthly bundle",
        "reference_provider": "existing validated BaoStock cache (read-only comparison)",
        "reference_directory": str(directory.relative_to(root)),
        "reference_security_files": len(files),
        "reference_manifest_sha256": sha256(
            "\n".join(f"{path.name}:{sha256_file(path)}" for path in files).encode("utf-8")
        ).hexdigest(),
        "shared_securities": len(shared_codes),
        "matched_rows": len(merged),
        "matched_start_date": merged["date"].min().date().isoformat(),
        "matched_end_date": merged["date"].max().date().isoformat(),
        "open_consistency": open_stats,
        "close_consistency": close_stats,
        "preclose_consistency": preclose_stats,
        "tradestatus_suspension_agreement": suspension_agreement,
        "sign_close_minus_open_agreement": open_close_sign,
        "sign_close_minus_preclose_agreement": close_preclose_sign,
        "emotion_smoothed_sign_agreement": emotion_sign_agreement,
        "emotion_smoothed_correlation": emotion_correlation,
        "relative_50_500_signal_agreement": signal_agreement,
        "emotion_comparison_rows": len(emotion),
        "signal_comparison_rows": len(signals),
        "checks": checks,
        "difference_explanation": {
            "rounding": "The providers can differ by one quotation tick because source rounding is independent.",
            "corporate_actions": "Production preclose is audited from prior raw close, RQAlpha cash/split records and cumulative factors; the bundle reference is used only for classified complex-action or suspension-reset exceptions.",
            "suspensions": "RQAlpha status comes only from suspended_days.h5; BaoStock tradestatus is used only as an external comparison label.",
            "coverage": "Emotion and signals use the common-code sample with official PIT membership; they are validation statistics, not production inputs.",
        },
    }
    audit_path = root / "audit" / "migration_validation.json"
    write_json(audit_path, result)
    if result["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Data migration validation failed: {failed}")
    return result

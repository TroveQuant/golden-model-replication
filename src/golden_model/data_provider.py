"""Provider-neutral data contracts and cache integrity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MarketDataBundle:
    """Validated observations required by the strategy and backtest."""

    trade_dates: pd.DataFrame
    constituents: pd.DataFrame
    stock_prices: pd.DataFrame
    index_prices: pd.DataFrame
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_market_data(bundle: MarketDataBundle, config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on missing, duplicated, future-dated, or malformed source data."""

    required_trade = {"date", "is_trading_day"}
    required_constituents = {
        "snapshot_date",
        "provider_update_date",
        "universe",
        "code",
        "code_name",
    }
    required_prices = {
        "date",
        "code",
        "open",
        "close",
        "preclose",
        "turn",
        "tradestatus",
        "security_status",
    }
    frames = {
        "trade_dates": (bundle.trade_dates, required_trade),
        "constituents": (bundle.constituents, required_constituents),
        "stock_prices": (bundle.stock_prices, required_prices),
        "index_prices": (bundle.index_prices, required_prices),
    }
    for name, (frame, required) in frames.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} is empty; no fallback data are permitted.")

    for frame_name in ("trade_dates", "stock_prices", "index_prices"):
        frame = getattr(bundle, frame_name)
        if frame["date"].isna().any():
            raise ValueError(f"{frame_name} contains invalid dates.")

    constituents = bundle.constituents
    if (constituents["provider_update_date"] > constituents["snapshot_date"]).any():
        raise ValueError("Constituent data contain a future provider update date.")
    if constituents.duplicated(["snapshot_date", "universe", "code"]).any():
        raise ValueError("Constituent snapshots contain duplicate codes.")
    if bundle.stock_prices.duplicated(["date", "code"]).any():
        raise ValueError("Stock prices contain duplicate date/code observations.")
    if bundle.index_prices.duplicated(["date", "code"]).any():
        raise ValueError("Index prices contain duplicate date/code observations.")

    constituent_codes = set(constituents["code"].astype(str))
    observed_stock_codes = set(bundle.stock_prices["code"].astype(str))
    missing_constituent_prices = sorted(constituent_codes - observed_stock_codes)
    if missing_constituent_prices:
        preview = missing_constituent_prices[:10]
        raise ValueError(
            "Historical constituents are missing all stock-price observations; "
            f"count={len(missing_constituent_prices)}, examples={preview}."
        )
    expected_history = int(config["data"].get("expected_historical_securities", 0))
    if expected_history and (
        len(constituent_codes) != expected_history
        or len(observed_stock_codes) != expected_history
    ):
        raise ValueError(
            "Historical security coverage must match the audited official event chain; "
            f"expected={expected_history}, constituents={len(constituent_codes)}, "
            f"prices={len(observed_stock_codes)}."
        )

    universes = config["data"]["universes"]
    counts = (
        constituents.groupby(["snapshot_date", "universe"], observed=True)["code"]
        .nunique()
        .rename("count")
        .reset_index()
    )
    for universe, definition in universes.items():
        subset = counts.loc[counts["universe"] == universe, "count"]
        expected = int(definition["expected_constituents"])
        if subset.empty or not subset.eq(expected).all():
            actual = None if subset.empty else int(subset.min())
            raise ValueError(
                f"{universe} constituent count must always equal {expected}; observed minimum={actual}."
            )

    requested_start = pd.Timestamp(config["data"]["start_date"])
    requested_end = pd.Timestamp(config["data"]["end_date"])
    effective_end = pd.Timestamp(config["data"].get("effective_end_date", requested_end))
    trading_days = set(
        bundle.trade_dates.loc[
            bundle.trade_dates["is_trading_day"].astype(int) == 1, "date"
        ]
    )
    if effective_end not in trading_days:
        raise ValueError(
            f"Effective end date {effective_end.date().isoformat()} is not a trading day."
        )
    stock_dates = bundle.stock_prices["date"]
    index_dates = bundle.index_prices["date"]
    if stock_dates.max() < effective_end or index_dates.max() < effective_end:
        raise ValueError(
            "Provider data do not reach the effective end date; "
            f"stock_max={stock_dates.max().date().isoformat()}, "
            f"index_max={index_dates.max().date().isoformat()}, "
            f"required={effective_end.date().isoformat()}."
        )
    if stock_dates.min() > requested_start or index_dates.min() > requested_start:
        raise ValueError("Provider data begin after the configured strategy start date.")

    invalid_prices = (
        bundle.index_prices[["open", "close", "preclose"]].le(0).any(axis=1)
        | bundle.index_prices[["open", "close", "preclose"]].isna().any(axis=1)
    )
    if invalid_prices.any():
        raise ValueError("Execution-index prices contain missing or non-positive values.")

    allowed_status = {"pre_listing", "suspended", "trading", "post_delisting", "missing"}
    observed_status = set(bundle.stock_prices["security_status"].dropna().astype(str))
    if not observed_status.issubset(allowed_status):
        raise ValueError(f"Unknown RQAlpha security statuses: {sorted(observed_status - allowed_status)}")
    inconsistent = bundle.stock_prices[
        (bundle.stock_prices["security_status"].eq("suspended"))
        != bundle.stock_prices["tradestatus"].eq(0)
    ]
    if not inconsistent.empty:
        raise ValueError("tradestatus does not agree with RQAlpha suspended_days classification.")

    return {
        "trade_date_rows": int(len(bundle.trade_dates)),
        "constituent_rows": int(len(bundle.constituents)),
        "stock_price_rows": int(len(bundle.stock_prices)),
        "index_price_rows": int(len(bundle.index_prices)),
        "unique_stocks": int(bundle.stock_prices["code"].nunique()),
        "historical_constituent_codes": int(len(constituent_codes)),
        "stock_price_missing_ohlc_rows": int(
            bundle.stock_prices[["open", "close", "preclose"]]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "stock_price_nontrading_rows": int(
            bundle.stock_prices["tradestatus"].fillna(0).astype(int).ne(1).sum()
        ),
        "actual_start": min(stock_dates.min(), index_dates.min()).date().isoformat(),
        "requested_end": requested_end.date().isoformat(),
        "effective_end": effective_end.date().isoformat(),
        "actual_end": min(stock_dates.max(), index_dates.max()).date().isoformat(),
    }

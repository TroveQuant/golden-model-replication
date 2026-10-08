from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from golden_model.data_provider import MarketDataBundle, validate_market_data


def _bundle() -> MarketDataBundle:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    trade_dates = pd.DataFrame({"date": dates, "is_trading_day": [1, 1]})
    constituents = []
    for universe, size in (("sse50", 50), ("csi500", 500)):
        for number in range(size):
            constituents.append(
                {
                    "snapshot_date": dates[0],
                    "provider_update_date": dates[0],
                    "universe": universe,
                    "code": f"{universe}.{number:04d}",
                    "code_name": str(number),
                }
            )
    member_codes = [row["code"] for row in constituents]
    stock_prices = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": 1.0,
                "close": 1.0,
                "preclose": 1.0,
                "turn": 1.0,
                "tradestatus": 1,
                "security_status": "trading",
            }
            for code in member_codes
            for date in dates
        ]
    )
    index_prices = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": 1.0,
                "close": 1.0,
                "preclose": 1.0,
                "turn": 1.0,
                "tradestatus": 1,
                "security_status": "trading",
            }
            for code in ("sh.000016", "sh.000905")
            for date in dates
        ]
    )
    return MarketDataBundle(
        trade_dates,
        pd.DataFrame(constituents),
        stock_prices,
        index_prices,
        {},
    )


def _config(project_config: dict) -> dict:
    config = deepcopy(project_config)
    config["data"]["start_date"] = "2024-01-02"
    config["data"]["end_date"] = "2024-01-03"
    config["data"]["expected_historical_securities"] = 550
    return config


def test_expected_historical_security_count_is_fail_closed(project_config: dict) -> None:
    config = _config(project_config)
    config["data"]["expected_historical_securities"] = 551
    with pytest.raises(ValueError, match="Historical security coverage"):
        validate_market_data(_bundle(), config)


def test_valid_data_bundle_passes(project_config: dict) -> None:
    summary = validate_market_data(_bundle(), _config(project_config))
    assert summary["actual_end"] == "2024-01-03"


def test_future_constituent_update_is_rejected(project_config: dict) -> None:
    bundle = _bundle()
    bad = bundle.constituents.copy()
    bad.loc[0, "provider_update_date"] = pd.Timestamp("2024-01-04")
    invalid = MarketDataBundle(
        bundle.trade_dates, bad, bundle.stock_prices, bundle.index_prices, {}
    )
    with pytest.raises(ValueError, match="future provider update"):
        validate_market_data(invalid, _config(project_config))


def test_duplicate_stock_observation_is_rejected(project_config: dict) -> None:
    bundle = _bundle()
    duplicated = pd.concat(
        [bundle.stock_prices, bundle.stock_prices.iloc[[0]]], ignore_index=True
    )
    invalid = MarketDataBundle(
        bundle.trade_dates, bundle.constituents, duplicated, bundle.index_prices, {}
    )
    with pytest.raises(ValueError, match="duplicate date/code"):
        validate_market_data(invalid, _config(project_config))


def test_missing_constituent_prices_are_rejected(project_config: dict) -> None:
    bundle = _bundle()
    missing_code = str(bundle.constituents.iloc[0]["code"])
    incomplete = bundle.stock_prices.loc[bundle.stock_prices["code"] != missing_code]
    invalid = MarketDataBundle(
        bundle.trade_dates, bundle.constituents, incomplete, bundle.index_prices, {}
    )
    with pytest.raises(ValueError, match="missing all stock-price observations"):
        validate_market_data(invalid, _config(project_config))


def test_stale_provider_data_are_rejected(project_config: dict) -> None:
    bundle = _bundle()
    stale_stock = bundle.stock_prices.loc[
        bundle.stock_prices["date"] < pd.Timestamp("2024-01-03")
    ]
    invalid = MarketDataBundle(
        bundle.trade_dates, bundle.constituents, stale_stock, bundle.index_prices, {}
    )
    with pytest.raises(ValueError, match="do not reach the effective end date"):
        validate_market_data(invalid, _config(project_config))

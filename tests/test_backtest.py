from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from golden_model.backtest import run_backtest


def _prices() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    rows = []
    for code, closes in {
        "sh.000016": [100.0, 110.0, 121.0, 133.1],
        "sh.000905": [100.0, 100.0, 100.0, 90.0],
    }.items():
        for date, close in zip(dates, closes):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": close,
                    "close": close,
                    "preclose": close,
                    "turn": 1.0,
                    "tradestatus": 1,
                }
            )
    return pd.DataFrame(rows)


def _config(project_config: dict) -> dict:
    config = deepcopy(project_config)
    config["data"]["start_date"] = "2024-01-02"
    config["data"]["end_date"] = "2024-01-04"
    return config


def test_signal_at_t_executes_at_t_plus_one_close(project_config: dict) -> None:
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "model": ["model2"],
            "sse50_weight": [0.4],
            "csi500_weight": [-0.4],
            "cash_sleeve": [0.2],
        }
    )
    result = run_backtest(_prices(), targets, _config(project_config))
    curve = result.equity_curve.loc[result.equity_curve["model"] == "model2"]
    jan3 = curve.loc[curve["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    jan4 = curve.loc[curve["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert jan3["gross_return"] == 0.0  # The new close execution earns no prior interval.
    assert jan3["transaction_cost"] == pytest.approx(0.8 * 0.0005)
    assert jan4["gross_return"] == pytest.approx(0.4 * 0.10 - 0.4 * -0.10)
    trade = result.trades.iloc[0]
    assert trade["signal_date"] == pd.Timestamp("2024-01-02")
    assert trade["execution_date"] == pd.Timestamp("2024-01-03")


def test_reversal_cost_uses_sum_of_absolute_weight_changes(project_config: dict) -> None:
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "model": ["model2", "model2"],
            "sse50_weight": [0.4, -0.4],
            "csi500_weight": [-0.4, 0.4],
            "cash_sleeve": [0.2, 0.2],
        }
    )
    result = run_backtest(_prices(), targets, _config(project_config))
    positions = result.positions.loc[result.positions["model"] == "model2"]
    jan4 = positions.loc[positions["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    curve = result.equity_curve.loc[
        (result.equity_curve["model"] == "model2")
        & (result.equity_curve["date"] == pd.Timestamp("2024-01-04"))
    ].iloc[0]
    assert jan4["turnover"] == pytest.approx(1.6)
    assert curve["transaction_cost"] == pytest.approx(1.6 * 0.0005)


def test_next_session_execution_crosses_a_weekend(project_config: dict) -> None:
    prices = _prices().copy()
    remap = {
        pd.Timestamp("2024-01-01"): pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-02"): pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-03"): pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-04"): pd.Timestamp("2024-01-09"),
    }
    prices["date"] = prices["date"].map(remap)
    config = deepcopy(project_config)
    config["data"]["start_date"] = "2024-01-05"
    config["data"]["end_date"] = "2024-01-09"
    targets = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05")],
            "model": ["model2"],
            "sse50_weight": [0.4],
            "csi500_weight": [-0.4],
            "cash_sleeve": [0.2],
        }
    )
    result = run_backtest(prices, targets, config)
    assert result.trades["execution_date"].min() == pd.Timestamp("2024-01-08")

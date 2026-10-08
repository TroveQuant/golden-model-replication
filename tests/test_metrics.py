from __future__ import annotations

import math

import pandas as pd
import pytest

from golden_model.metrics import annualized_sharpe, calculate_metrics, maximum_drawdown


def test_maximum_drawdown() -> None:
    nav = pd.Series([1.0, 0.8, 0.9, 0.7, 1.1])
    assert maximum_drawdown(nav) == pytest.approx(-0.30)


def test_sharpe_matches_sample_standard_deviation_formula() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, 0.0])
    expected = math.sqrt(252) * returns.mean() / returns.std(ddof=1)
    assert annualized_sharpe(returns, 252, 0.0) == pytest.approx(expected)


def test_excess_metrics_use_the_same_date_aligned_benchmark() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    equity = pd.DataFrame(
        [
            {
                "date": date,
                "model": model,
                "daily_return": daily,
                "nav": nav,
                "transaction_cost": 0.0,
            }
            for model, returns, navs in (
                ("model2", [0.0, 0.10, 0.0], [1.0, 1.10, 1.10]),
                ("benchmark_equal_weight", [0.0, 0.05, 0.0], [1.0, 1.05, 1.05]),
            )
            for date, daily, nav in zip(dates, returns, navs)
        ]
    )
    positions = pd.DataFrame(columns=["model", "turnover"])
    trades = pd.DataFrame(columns=["model"])
    metrics = calculate_metrics(equity, positions, trades)
    assert metrics["model2"]["benchmark_total_return"] == pytest.approx(0.05)
    assert metrics["model2"]["excess_total_return"] == pytest.approx(1.10 / 1.05 - 1)
    assert metrics["model2"]["tracking_error"] is not None

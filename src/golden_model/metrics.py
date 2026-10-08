"""Deterministic performance and risk metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def drawdown_series(nav: pd.Series) -> pd.Series:
    values = pd.to_numeric(nav, errors="raise")
    return values / values.cummax() - 1.0


def maximum_drawdown(nav: pd.Series) -> float:
    return float(drawdown_series(nav).min())


def maximum_drawdown_duration(nav: pd.Series) -> int:
    underwater = drawdown_series(nav) < 0
    longest = current = 0
    for value in underwater:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return int(longest)


def annualized_sharpe(
    returns: pd.Series, annualization_days: int = 252, risk_free_rate: float = 0.0
) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if len(values) < 2:
        return None
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / annualization_days) - 1.0
    excess = values - daily_risk_free
    volatility = float(excess.std(ddof=1))
    if volatility == 0 or not math.isfinite(volatility):
        return None
    return float(math.sqrt(annualization_days) * excess.mean() / volatility)


def calculate_metrics(
    equity_curve: pd.DataFrame,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
    initial_nav: float = 1.0,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    benchmark_name = "benchmark_equal_weight"
    benchmark_frame = equity_curve.loc[
        equity_curve["model"] == benchmark_name
    ].sort_values("date")
    benchmark_return_by_date = (
        benchmark_frame.set_index("date")["daily_return"]
        if not benchmark_frame.empty
        else pd.Series(dtype="float64")
    )
    benchmark_final_nav = (
        float(benchmark_frame["nav"].iloc[-1])
        if not benchmark_frame.empty
        else None
    )
    for model, frame in equity_curve.groupby("model", sort=True):
        frame = frame.sort_values("date")
        returns = frame["daily_return"]
        nav = frame["nav"]
        periods = max(len(frame) - 1, 1)
        final_nav = float(nav.iloc[-1])
        total_return = final_nav / initial_nav - 1.0
        annual_return = (final_nav / initial_nav) ** (
            annualization_days / periods
        ) - 1.0
        annual_volatility = float(returns.std(ddof=1) * math.sqrt(annualization_days))
        model_positions = positions.loc[positions["model"] == model]
        model_trades = trades.loc[trades["model"] == model]
        max_drawdown = maximum_drawdown(nav)
        calmar = (
            float(annual_return / abs(max_drawdown))
            if max_drawdown < 0
            else None
        )
        benchmark_total_return: float | None = None
        excess_total_return: float | None = None
        annualized_excess_return: float | None = None
        tracking_error: float | None = None
        information_ratio: float | None = None
        if benchmark_final_nav is not None:
            benchmark_total_return = benchmark_final_nav / initial_nav - 1.0
            excess_total_return = final_nav / benchmark_final_nav - 1.0
            annualized_excess_return = (final_nav / benchmark_final_nav) ** (
                annualization_days / periods
            ) - 1.0
            aligned = frame.set_index("date")["daily_return"].align(
                benchmark_return_by_date, join="inner"
            )
            excess_daily = aligned[0] - aligned[1]
            if len(excess_daily) >= 2:
                daily_tracking_error = float(excess_daily.std(ddof=1))
                if daily_tracking_error > 0 and math.isfinite(daily_tracking_error):
                    tracking_error = daily_tracking_error * math.sqrt(annualization_days)
                    information_ratio = float(
                        math.sqrt(annualization_days)
                        * excess_daily.mean()
                        / daily_tracking_error
                    )

        output[str(model)] = {
            "start_date": pd.Timestamp(frame["date"].min()).date().isoformat(),
            "end_date": pd.Timestamp(frame["date"].max()).date().isoformat(),
            "observations": int(len(frame)),
            "initial_nav": float(initial_nav),
            "final_nav": final_nav,
            "total_return": float(total_return),
            "annualized_return": float(annual_return),
            "annualized_volatility": annual_volatility,
            "sharpe": annualized_sharpe(
                returns, annualization_days, risk_free_rate
            ),
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_sessions": maximum_drawdown_duration(nav),
            "benchmark_total_return": benchmark_total_return,
            "excess_total_return": excess_total_return,
            "annualized_excess_return": annualized_excess_return,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "total_turnover": float(model_positions["turnover"].sum())
            if not model_positions.empty
            else 0.0,
            "total_transaction_cost": float(frame["transaction_cost"].sum()),
            "trade_legs": int(len(model_trades)),
        }
    return output

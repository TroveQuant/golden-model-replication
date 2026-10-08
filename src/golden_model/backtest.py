"""Close-to-close backtest with strictly delayed next-close execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .metrics import calculate_metrics, drawdown_series


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, dict[str, Any]]


def _price_table(index_prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    code_to_asset = {
        code: asset for asset, code in config["data"]["execution_assets"].items()
    }
    frame = index_prices.loc[index_prices["code"].isin(code_to_asset)].copy()
    frame["asset"] = frame["code"].map(code_to_asset)
    table = frame.pivot(index="date", columns="asset", values="close").sort_index()
    required = ["sse50", "csi500"]
    if table[required].isna().any().any():
        raise ValueError("Execution assets do not share a complete trading calendar.")
    return table[required]


def run_backtest(
    index_prices: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
) -> BacktestResult:
    price = _price_table(index_prices, config)
    start = pd.Timestamp(config["data"]["start_date"])
    end = pd.Timestamp(config["data"].get("effective_end_date", config["data"]["end_date"]))
    dates = price.index[(price.index >= start) & (price.index <= end)]
    if len(dates) < 2:
        raise ValueError("Backtest requires at least two execution sessions.")
    full_dates = list(price.index)
    date_position = {date: index for index, date in enumerate(full_dates)}
    returns = price.pct_change(fill_method=None)
    cost_rate = float(config["backtest"]["one_way_cost_bps"]) / 10000.0
    initial_nav = float(config["backtest"]["initial_nav"])

    equity_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for model, model_targets in targets.groupby("model", sort=True):
        target_lookup = model_targets.set_index("date").sort_index()
        held = {"sse50": 0.0, "csi500": 0.0}
        nav = initial_nav
        for output_index, date in enumerate(dates):
            price_index = date_position[date]
            prior_date = full_dates[price_index - 1] if price_index > 0 else None
            if output_index == 0:
                asset_returns = {"sse50": 0.0, "csi500": 0.0}
            else:
                asset_returns = {
                    "sse50": float(returns.at[date, "sse50"]),
                    "csi500": float(returns.at[date, "csi500"]),
                }
            gross_return = sum(held[asset] * asset_returns[asset] for asset in held)
            pre_trade = dict(held)
            target_weights = dict(held)
            signal_date: pd.Timestamp | None = None
            cash_sleeve = 1.0 if not any(held.values()) else float(
                config["strategy"]["weights"]["cash"]
            )
            if prior_date is not None and prior_date in target_lookup.index:
                target_row = target_lookup.loc[prior_date]
                if isinstance(target_row, pd.DataFrame):
                    raise ValueError(f"Duplicate targets for {model} on {prior_date}.")
                target_weights = {
                    "sse50": float(target_row["sse50_weight"]),
                    "csi500": float(target_row["csi500_weight"]),
                }
                cash_sleeve = float(target_row["cash_sleeve"])
                signal_date = prior_date
            deltas = {
                asset: target_weights[asset] - pre_trade[asset]
                for asset in target_weights
            }
            turnover = sum(abs(value) for value in deltas.values())
            transaction_cost = turnover * cost_rate
            daily_return = gross_return - transaction_cost
            nav *= 1.0 + daily_return
            held = target_weights

            equity_rows.append(
                {
                    "date": date,
                    "model": model,
                    "gross_return": gross_return,
                    "transaction_cost": transaction_cost,
                    "daily_return": daily_return,
                    "nav": nav,
                }
            )
            position_rows.append(
                {
                    "date": date,
                    "model": model,
                    "signal_date": signal_date,
                    "execution_date": date if signal_date is not None else pd.NaT,
                    "pre_sse50_weight": pre_trade["sse50"],
                    "pre_csi500_weight": pre_trade["csi500"],
                    "sse50_weight": held["sse50"],
                    "csi500_weight": held["csi500"],
                    "cash_sleeve": cash_sleeve,
                    "gross_exposure": abs(held["sse50"]) + abs(held["csi500"]),
                    "net_exposure": held["sse50"] + held["csi500"],
                    "turnover": turnover,
                }
            )
            if turnover > 0:
                for asset, delta in deltas.items():
                    if delta != 0:
                        trade_rows.append(
                            {
                                "signal_date": signal_date,
                                "execution_date": date,
                                "model": model,
                                "asset": asset,
                                "from_weight": pre_trade[asset],
                                "to_weight": held[asset],
                                "weight_change": delta,
                                "one_way_cost_bps": cost_rate * 10000.0,
                                "transaction_cost": abs(delta) * cost_rate,
                            }
                        )

    benchmark_nav = initial_nav
    for index, date in enumerate(dates):
        benchmark_return = 0.0
        if index > 0:
            benchmark_return = float(
                0.5 * returns.at[date, "sse50"] + 0.5 * returns.at[date, "csi500"]
            )
        benchmark_nav *= 1.0 + benchmark_return
        equity_rows.append(
            {
                "date": date,
                "model": "benchmark_equal_weight",
                "gross_return": benchmark_return,
                "transaction_cost": 0.0,
                "daily_return": benchmark_return,
                "nav": benchmark_nav,
            }
        )

    equity = pd.DataFrame(equity_rows).sort_values(["model", "date"])
    equity["drawdown"] = equity.groupby("model", observed=True)["nav"].transform(
        drawdown_series
    )
    positions = pd.DataFrame(position_rows).sort_values(["model", "date"])
    trade_columns = [
        "signal_date",
        "execution_date",
        "model",
        "asset",
        "from_weight",
        "to_weight",
        "weight_change",
        "one_way_cost_bps",
        "transaction_cost",
    ]
    trades = pd.DataFrame(trade_rows, columns=trade_columns).sort_values(
        ["model", "execution_date", "asset"]
    )
    metrics = calculate_metrics(
        equity,
        positions,
        trades,
        annualization_days=int(config["backtest"]["annualization_days"]),
        risk_free_rate=float(config["backtest"]["risk_free_rate"]),
        initial_nav=initial_nav,
    )
    return BacktestResult(equity, positions, trades, metrics)

"""Public-data proxy implementations of the two Golden Model sentiment factors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TEN_BUCKET_WEIGHTS: dict[int, float] = {
    5: 50.33223,
    4: 25.0,
    3: 8.3,
    2: 4.6342,
    1: 0.7465,
    -1: -0.873,
    -2: -4.234,
    -3: -9.88764,
    -4: -25.0,
    -5: -50.8763,
}


def continuous_signed_streak(values: pd.Series, window: int = 5) -> pd.Series:
    """Count consecutive positive/negative observations; zero and NaN reset to zero."""

    signs = np.sign(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy())
    output = np.zeros(len(signs), dtype=np.int64)
    previous = 0
    for index, sign in enumerate(signs):
        current = int(sign)
        if current == 0:
            previous = 0
        elif previous == 0 or np.sign(previous) != current:
            previous = current
        else:
            previous = int(np.clip(previous + current, -window, window))
        output[index] = previous
    return pd.Series(output, index=values.index, dtype="int64")


def add_stock_factor_states(
    stock_prices: pd.DataFrame, state_window: int = 5
) -> pd.DataFrame:
    """Calculate both documented public-data proxy definitions per security."""

    frame = stock_prices.sort_values(["code", "date"]).copy()
    frame["model1_input"] = frame["close"] - frame["open"]
    frame["model2_input"] = frame["close"] - frame["preclose"]
    for model in ("model1", "model2"):
        state_column = f"{model}_state"
        weight_column = f"{model}_weight"
        frame[state_column] = (
            frame.groupby("code", sort=False, group_keys=False)[f"{model}_input"]
            .apply(lambda values: continuous_signed_streak(values, state_window))
            .astype(int)
        )
        frame[weight_column] = frame[state_column].map(TEN_BUCKET_WEIGHTS)
    return frame


def _membership_observations(
    factors: pd.DataFrame, constituents: pd.DataFrame, universe: str
) -> pd.DataFrame:
    snapshots = constituents.loc[constituents["universe"] == universe].copy()
    if snapshots.empty:
        raise ValueError(f"No point-in-time constituents for {universe}.")
    snapshot_dates = sorted(pd.to_datetime(snapshots["snapshot_date"].unique()))
    factors_by_date = factors.sort_values("date").set_index("date", drop=False)
    pieces: list[pd.DataFrame] = []
    for index, snapshot_date in enumerate(snapshot_dates):
        next_date = snapshot_dates[index + 1] if index + 1 < len(snapshot_dates) else None
        members = snapshots.loc[snapshots["snapshot_date"] == snapshot_date, "code"]
        if next_date is None:
            interval = factors_by_date.loc[snapshot_date:].copy()
        else:
            interval = factors_by_date.loc[snapshot_date:next_date].copy()
            interval = interval.loc[interval["date"] < next_date]
        selected = interval.loc[interval["code"].isin(members)].copy()
        if not selected.empty:
            selected["universe"] = universe
            selected["snapshot_date"] = snapshot_date
            pieces.append(selected)
    if not pieces:
        raise ValueError(f"No price observations matched constituents for {universe}.")
    output = pd.concat(pieces, ignore_index=True)
    if (output["snapshot_date"] > output["date"]).any():
        raise ValueError("Future constituent snapshot leaked into factor observations.")
    return output


def calculate_sentiment(
    stock_prices: pd.DataFrame,
    constituents: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build daily model 1/model 2 universe sentiment and three-day smoothing."""

    strategy = config["strategy"]
    factors = add_stock_factor_states(stock_prices, int(strategy["state_window"]))
    memberships = [
        _membership_observations(factors, constituents, universe)
        for universe in config["data"]["universes"]
    ]
    observations = pd.concat(memberships, ignore_index=True)
    observations = observations.loc[observations["tradestatus"] == 1].copy()

    results: list[pd.DataFrame] = []
    for model in ("model1", "model2"):
        state_column = f"{model}_state"
        weight_column = f"{model}_weight"
        eligible = observations.loc[
            observations[state_column].ne(0) & observations[weight_column].notna()
        ]
        daily = (
            eligible.groupby(["date", "universe"], observed=True)
            .agg(
                raw_score=(weight_column, "mean"),
                eligible_count=("code", "nunique"),
                snapshot_date=("snapshot_date", "max"),
            )
            .reset_index()
            .sort_values(["universe", "date"])
        )
        daily["smoothed_score"] = daily.groupby("universe", observed=True)[
            "raw_score"
        ].transform(
            lambda values: values.rolling(
                int(strategy["rolling_window"]),
                min_periods=int(strategy["rolling_window"]),
            ).mean()
        )
        daily["model"] = model
        results.append(daily)
    output = pd.concat(results, ignore_index=True)
    return output[
        [
            "date",
            "model",
            "universe",
            "snapshot_date",
            "eligible_count",
            "raw_score",
            "smoothed_score",
        ]
    ].sort_values(["model", "date", "universe"]).reset_index(drop=True)

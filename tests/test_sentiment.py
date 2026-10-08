from __future__ import annotations

import pandas as pd
import pytest

from golden_model.sentiment import (
    TEN_BUCKET_WEIGHTS,
    _membership_observations,
    add_stock_factor_states,
    calculate_sentiment,
    continuous_signed_streak,
)


def test_continuous_red_green_count_caps_and_resets() -> None:
    values = pd.Series([1, 2, 3, 4, 5, 6, 0, -1, -2, -3, -4, -5, -6, 2])
    actual = continuous_signed_streak(values, window=5).tolist()
    assert actual == [1, 2, 3, 4, 5, 5, 0, -1, -2, -3, -4, -5, -5, 1]


def test_ten_bucket_weights_are_exact() -> None:
    assert len(TEN_BUCKET_WEIGHTS) == 10
    assert TEN_BUCKET_WEIGHTS == {
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


def test_model_factor_formulas_use_open_and_preclose() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["sh.600000", "sh.600000"],
            "open": [10.0, 12.0],
            "close": [11.0, 11.0],
            "preclose": [12.0, 11.0],
            "turn": [1.0, 1.0],
            "tradestatus": [1, 1],
        }
    )
    result = add_stock_factor_states(prices)
    assert result["model1_input"].tolist() == [1.0, -1.0]
    assert result["model2_input"].tolist() == [-1.0, 0.0]
    assert result["model1_state"].tolist() == [1, -1]
    assert result["model2_state"].tolist() == [-1, 0]


def test_membership_snapshot_never_applies_before_snapshot_date() -> None:
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "code": ["A", "B", "B"],
        }
    )
    constituents = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "universe": ["sse50", "sse50"],
            "code": ["A", "B"],
        }
    )
    result = _membership_observations(factors, constituents, "sse50")
    assert list(zip(result["date"], result["code"])) == [
        (pd.Timestamp("2024-01-02"), "A"),
        (pd.Timestamp("2024-01-04"), "B"),
    ]
    assert (result["snapshot_date"] <= result["date"]).all()


def test_sentiment_uses_three_complete_trading_observations(project_config: dict) -> None:
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    prices = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": 10.0,
                "close": close,
                "preclose": 10.0,
                "turn": 1.0,
                "tradestatus": 1,
            }
            for code, closes in (("A", [11.0] * 4), ("B", [9.0] * 4))
            for date, close in zip(dates, closes)
        ]
    )
    constituents = pd.DataFrame(
        {
            "snapshot_date": [dates[0], dates[0]],
            "universe": ["sse50", "csi500"],
            "code": ["A", "B"],
        }
    )
    result = calculate_sentiment(prices, constituents, project_config)
    model1_sse = result.loc[
        (result["model"] == "model1") & (result["universe"] == "sse50")
    ].reset_index(drop=True)
    assert model1_sse["smoothed_score"].iloc[:2].isna().all()
    assert model1_sse["smoothed_score"].iloc[2] == pytest.approx(
        model1_sse["raw_score"].iloc[:3].mean()
    )

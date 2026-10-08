"""Translate relative-strength directions into explicit target weights."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_target_weights(signals: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    weights = config["strategy"]["weights"]
    long_weight = float(weights["sse50_long"])
    short_weight = float(weights["csi500_short"])
    active_cash = float(weights["cash"])
    if long_weight <= 0 or short_weight >= 0:
        raise ValueError("Configured long/short weights have invalid signs.")

    output = signals[["date", "model", "direction", "signal_value"]].copy()
    output["sse50_weight"] = 0.0
    output["csi500_weight"] = 0.0
    positive = output["direction"] > 0
    negative = output["direction"] < 0
    output.loc[positive, "sse50_weight"] = long_weight
    output.loc[positive, "csi500_weight"] = short_weight
    output.loc[negative, "sse50_weight"] = short_weight
    output.loc[negative, "csi500_weight"] = long_weight
    output["cash_sleeve"] = active_cash
    output.loc[output["direction"] == 0, "cash_sleeve"] = 1.0
    output["gross_exposure"] = (
        output["sse50_weight"].abs() + output["csi500_weight"].abs()
    )
    output["net_exposure"] = output["sse50_weight"] + output["csi500_weight"]
    return output

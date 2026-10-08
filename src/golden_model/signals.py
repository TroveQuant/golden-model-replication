"""Relative-strength signal construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_signals(sentiment: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if config["strategy"]["signal_rule"] != "delta_sse50_minus_delta_csi500":
        raise ValueError("Unsupported signal rule.")
    outputs: list[pd.DataFrame] = []
    for model, model_frame in sentiment.groupby("model", sort=True):
        wide = model_frame.pivot(
            index="date", columns="universe", values="smoothed_score"
        ).sort_index()
        for required in ("sse50", "csi500"):
            if required not in wide:
                raise ValueError(f"Missing {required} sentiment for {model}.")
        signal_value = wide["sse50"].diff() - wide["csi500"].diff()
        raw_direction = pd.Series(
            np.sign(signal_value), index=signal_value.index, dtype="float64"
        )
        if config["strategy"]["zero_signal_policy"] == "hold":
            direction = raw_direction.replace(0.0, np.nan).ffill().fillna(0.0)
        else:
            direction = raw_direction.fillna(0.0)
        outputs.append(
            pd.DataFrame(
                {
                    "date": wide.index,
                    "model": model,
                    "sse50_sentiment": wide["sse50"],
                    "csi500_sentiment": wide["csi500"],
                    "signal_value": signal_value,
                    "raw_direction": raw_direction.fillna(0.0).astype(int),
                    "direction": direction.astype(int),
                }
            )
        )
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["model", "date"]
    ).reset_index(drop=True)

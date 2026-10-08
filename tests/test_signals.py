from __future__ import annotations

import pandas as pd

from golden_model.signals import generate_signals


def test_relative_strength_signal_uses_difference_of_daily_changes(
    project_config: dict,
) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    sentiment = pd.DataFrame(
        [
            {"date": date, "model": "model2", "universe": universe, "smoothed_score": value}
            for universe, values in {
                "sse50": [1.0, 2.0, 2.0, 2.0],
                "csi500": [1.0, 1.0, 2.0, 2.0],
            }.items()
            for date, value in zip(dates, values)
        ]
    )
    result = generate_signals(sentiment, project_config)
    assert result["signal_value"].iloc[1:4].tolist() == [1.0, -1.0, 0.0]
    assert result["direction"].tolist() == [0, 1, -1, -1]

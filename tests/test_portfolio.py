from __future__ import annotations

import pandas as pd

from golden_model.portfolio import build_target_weights


def test_target_weights_follow_direction(project_config: dict) -> None:
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "model": ["model2"] * 3,
            "direction": [1, -1, 0],
            "signal_value": [1.0, -1.0, 0.0],
        }
    )
    result = build_target_weights(signals, project_config)
    assert result[["sse50_weight", "csi500_weight"]].values.tolist() == [
        [0.4, -0.4],
        [-0.4, 0.4],
        [0.0, 0.0],
    ]
    # A negative relative-strength signal reverses which sleeve is long/short.
    assert result.loc[1, "sse50_weight"] == project_config["strategy"]["weights"]["csi500_short"]
    assert result.loc[1, "csi500_weight"] == project_config["strategy"]["weights"]["sse50_long"]
    assert result.loc[2, "cash_sleeve"] == 1.0

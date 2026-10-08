from __future__ import annotations

import pandas as pd

from golden_model.report import build_report_html


def test_report_is_self_contained_and_contains_audit_sections(project_config: dict) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    equity = pd.DataFrame(
        [
            {
                "date": date,
                "model": model,
                "nav": nav,
                "drawdown": min(nav - 1.0, 0.0),
            }
            for model, navs in (
                ("model1", [1.0, 1.01]),
                ("model2", [1.0, 0.99]),
                ("benchmark_equal_weight", [1.0, 1.0]),
            )
            for date, nav in zip(dates, navs)
        ]
    )
    positions = pd.DataFrame(
        [
            {
                "date": dates[-1],
                "model": model,
                "sse50_weight": 0.4,
                "csi500_weight": -0.4,
                "cash_sleeve": 0.2,
                "gross_exposure": 0.8,
                "net_exposure": 0.0,
            }
            for model in ("model1", "model2")
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "signal_date": dates[0],
                "execution_date": dates[1],
                "model": "model2",
                "asset": "sse50",
                "from_weight": 0.0,
                "to_weight": 0.4,
                "transaction_cost": 0.0002,
            }
        ]
    )
    metrics = {
        model: {
            "final_nav": float(frame["nav"].iloc[-1]),
            "total_return": float(frame["nav"].iloc[-1] - 1.0),
            "max_drawdown": float(frame["drawdown"].min()),
        }
        for model, frame in equity.groupby("model")
    }
    provenance = {
        "provider_version": "0.9.4",
        "retrieved_at_utc": "2024-01-03T10:00:00+00:00",
        "validation": {},
        "run": {"commit_sha": "test", "generated_at_utc": "test", "runtime": "test"},
    }
    html = build_report_html(
        project_config, equity, positions, trades, metrics, provenance
    )
    assert html.count("<svg") == 2
    assert "Methodology Mapping" in html
    assert "Fidelity Gaps and Limitations" in html
    assert "不构成投资建议" in html
    assert "http://" not in html and "https://" not in html

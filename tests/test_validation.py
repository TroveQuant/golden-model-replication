from __future__ import annotations

from copy import deepcopy

import pytest

from golden_model.migration_validation import _load_certified_audit
from golden_model.validation import validate_project_config


def test_production_rejects_mock_provider(project_config: dict, tmp_path) -> None:
    config = deepcopy(project_config)
    config["data"]["provider"] = "mock"
    with pytest.raises(ValueError, match="Forbidden or missing production data provider"):
        validate_project_config(config, tmp_path)


def test_certified_migration_audit_is_fail_closed(tmp_path) -> None:
    audit = tmp_path / "migration.json"
    audit.write_text(
        '{"status":"passed","production_provider":"RQAlpha free monthly bundle",'
        '"reference_security_files":1092,"matched_end_date":"2026-09-30",'
        '"checks":{"sign_close_minus_open":true,"sign_close_minus_preclose":true,'
        '"suspension":true,"relative_50_500_signal":true}}',
        encoding="utf-8",
    )
    settings = {"minimum_reference_securities": 1000}
    result = _load_certified_audit(audit, settings, "2026-09-30")
    assert result["gate_mode"] == "certified_read_only_audit"

    audit.write_text(
        audit.read_text(encoding="utf-8").replace(
            '"suspension":true', '"suspension":false'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a complete passing audit"):
        _load_certified_audit(audit, settings, "2026-09-30")

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def project_config() -> dict:
    with (ROOT / "config" / "default.yaml").open("r", encoding="utf-8") as handle:
        return deepcopy(yaml.safe_load(handle))

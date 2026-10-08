from __future__ import annotations

import numpy as np
import pytest

from golden_model.corporate_actions import construct_audited_preclose


BAR_DTYPE = np.dtype(
    [("datetime", "<i8"), ("close", "<f8"), ("prev_close", "<f8")]
)
FACTOR_DTYPE = np.dtype([("start_date", "<i8"), ("ex_cum_factor", "<f8")])
DIVIDEND_DTYPE = np.dtype(
    [("ex_dividend_date", "<i8"), ("dividend_cash_before_tax", "<f8"), ("round_lot", "<f8")]
)
SPLIT_DTYPE = np.dtype([("ex_date", "<i8"), ("split_factor", "<f8")])


def test_preclose_uses_prior_raw_close_on_ordinary_day() -> None:
    bars = np.array(
        [(20240102000000, 10.0, 9.8), (20240103000000, 10.5, 10.0)],
        dtype=BAR_DTYPE,
    )
    factors = np.array([(0, 1.0)], dtype=FACTOR_DTYPE)
    preclose, _, methods, audit = construct_audited_preclose(
        bars, factors, None, None, set()
    )
    assert preclose[1] == 10.0
    assert methods[1] == "prior_raw_close"
    assert audit["unexplained_difference_count"] == 0


def test_preclose_applies_cash_and_split_formula() -> None:
    bars = np.array(
        [(20240102000000, 99.0, 98.0), (20240103000000, 50.0, 49.0)],
        dtype=BAR_DTYPE,
    )
    factors = np.array([(0, 1.0), (20240103000000, 2.0)], dtype=FACTOR_DTYPE)
    dividends = np.array([(20240103, 10.0, 10.0)], dtype=DIVIDEND_DTYPE)
    splits = np.array([(20240103000000, 2.0)], dtype=SPLIT_DTYPE)
    preclose, _, methods, _ = construct_audited_preclose(
        bars, factors, dividends, splits, set()
    )
    assert preclose[1] == 49.0
    assert methods[1] == "cash_split_formula"


def test_suspension_reference_reset_is_explicitly_classified() -> None:
    bars = np.array(
        [(20180918000000, 52.69, 52.69), (20180919000000, 15.05, 15.05)],
        dtype=BAR_DTYPE,
    )
    factors = np.array([(0, 1.0)], dtype=FACTOR_DTYPE)
    preclose, _, methods, audit = construct_audited_preclose(
        bars, factors, None, None, {20180919}
    )
    assert preclose[1] == 15.05
    assert methods[1] == "bundle_reference:suspended_reference_reset"
    assert audit["controlled_fallback_count"] == 1


def test_unexplained_reference_difference_fails_closed() -> None:
    bars = np.array(
        [(20240102000000, 10.0, 10.0), (20240103000000, 9.0, 8.0)],
        dtype=BAR_DTYPE,
    )
    factors = np.array([(0, 1.0)], dtype=FACTOR_DTYPE)
    with pytest.raises(ValueError, match="Unexplained preclose"):
        construct_audited_preclose(bars, factors, None, None, set())

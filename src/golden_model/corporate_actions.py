"""Corporate-action-aware previous-close construction for RQAlpha bars."""

from __future__ import annotations

import numpy as np


def cumulative_factors_for_dates(
    bar_datetimes: np.ndarray, factor_records: np.ndarray | None
) -> np.ndarray:
    """Return RQAlpha's cumulative ex-factor applicable to every bar date."""

    if factor_records is None or len(factor_records) == 0:
        return np.ones(len(bar_datetimes), dtype="float64")
    starts = factor_records["start_date"].astype("int64", copy=False)
    factors = factor_records["ex_cum_factor"].astype("float64", copy=False)
    positions = np.searchsorted(starts, bar_datetimes, side="right") - 1
    if (positions < 0).any():
        raise ValueError("RQAlpha ex-factor series does not cover the first bar date.")
    result = factors[positions]
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("RQAlpha ex-factor series contains an invalid factor.")
    return result


def reconstruct_comparable_preclose(
    bars: np.ndarray, factor_records: np.ndarray | None
) -> np.ndarray:
    """Build each day's comparable preclose using RQAlpha's adjustment formula.

    This deliberately is not ``shift(close)``.  The previous raw close is
    multiplied by ``factor(previous_bar) / factor(current_bar)`` so dividends,
    splits, bonus issues and rights adjustments are expressed on the current
    day's price basis.  RQAlpha stores one daily bar for suspended sessions, so
    this also follows its official prior-trading-session semantics.
    """

    required = {"datetime", "close"}
    names = set(bars.dtype.names or ())
    if missing := required.difference(names):
        raise ValueError(f"Bars missing fields required for preclose: {sorted(missing)}")
    if len(bars) == 0:
        return np.empty(0, dtype="float64")
    datetimes = bars["datetime"].astype("int64", copy=False)
    if np.any(np.diff(datetimes) <= 0):
        raise ValueError("RQAlpha bars must be strictly ordered by datetime.")
    factors = cumulative_factors_for_dates(datetimes, factor_records)
    preclose = np.empty(len(bars), dtype="float64")
    # The first row has no prior bar in the extracted range.  Use the bundle's
    # source value only for that boundary observation; all later rows are rebuilt.
    if "prev_close" in names:
        preclose[0] = float(bars["prev_close"][0])
    else:
        preclose[0] = np.nan
    preclose[1:] = (
        bars["close"][:-1].astype("float64", copy=False)
        * factors[:-1]
        / factors[1:]
    )
    return preclose


def construct_audited_preclose(
    bars: np.ndarray,
    factor_records: np.ndarray | None,
    dividend_records: np.ndarray | None,
    split_records: np.ndarray | None,
    suspended_dates: set[int],
    *,
    absolute_tolerance: float = 0.011,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Construct an exact comparable preclose and audit every exception.

    RQAlpha's public API adjusts the previous raw close with the cumulative
    factor ratio; :func:`reconstruct_comparable_preclose` reproduces that
    behavior.  The free bundle also carries the exchange reference
    ``prev_close``.  It is more precise for compound rights/restructuring
    events and for resumptions where the suspended reference price was reset
    before the cumulative factor became effective.

    We therefore use three explicit, auditable paths:

    * ordinary sessions: prior raw close, which must match the reference;
    * simple cash/split sessions: the published cash and split formula;
    * complex corporate-action or suspension exceptions: the bundle's
      explicit comparable reference, only after an event/suspension condition
      has been proven.

    Any unexplained disagreement fails closed.
    """

    names = set(bars.dtype.names or ())
    required = {"datetime", "close", "prev_close"}
    if missing := required.difference(names):
        raise ValueError(f"Bars missing audited preclose fields: {sorted(missing)}")
    factor_based = reconstruct_comparable_preclose(bars, factor_records)
    datetimes = bars["datetime"].astype("int64", copy=False)
    dates = datetimes // 1_000_000
    reference = bars["prev_close"].astype("float64", copy=False)
    factors = cumulative_factors_for_dates(datetimes, factor_records)
    factor_changed = np.zeros(len(bars), dtype=bool)
    factor_changed[1:] = ~np.isclose(factors[1:], factors[:-1], rtol=0, atol=1e-12)

    cash_by_date: dict[int, float] = {}
    if dividend_records is not None:
        for record in dividend_records:
            round_lot = float(record["round_lot"])
            if not np.isfinite(round_lot) or round_lot <= 0:
                raise ValueError("RQAlpha dividend contains an invalid round lot.")
            date_value = int(record["ex_dividend_date"])
            cash_by_date[date_value] = cash_by_date.get(date_value, 0.0) + float(
                record["dividend_cash_before_tax"]
            ) / round_lot

    split_by_date: dict[int, float] = {}
    if split_records is not None:
        for record in split_records:
            date_value = int(record["ex_date"]) // 1_000_000
            factor = float(record["split_factor"])
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError("RQAlpha split data contains an invalid factor.")
            split_by_date[date_value] = split_by_date.get(date_value, 1.0) * factor

    output = factor_based.copy()
    methods = np.full(len(bars), "rqalpha_factor", dtype=object)
    methods[0] = "bundle_boundary"
    unresolved: list[dict[str, object]] = []
    fallback_examples: list[dict[str, object]] = []

    for position in range(1, len(bars)):
        date_value = int(dates[position])
        prior_date = int(dates[position - 1])
        current_suspended = date_value in suspended_dates
        prior_suspended = prior_date in suspended_dates
        has_explicit_action = date_value in cash_by_date or date_value in split_by_date

        if has_explicit_action:
            cash = cash_by_date.get(date_value, 0.0)
            split = split_by_date.get(date_value, 1.0)
            candidate = (float(bars["close"][position - 1]) - cash) / split
            candidate = float(np.round(candidate + 1e-12, 2))
            method = "cash_split_formula"
        elif not factor_changed[position]:
            candidate = float(bars["close"][position - 1])
            method = "prior_raw_close"
        else:
            candidate = float(factor_based[position])
            method = "rqalpha_factor"

        difference = abs(candidate - float(reference[position]))
        if difference <= absolute_tolerance:
            output[position] = candidate
            methods[position] = method
            continue

        if current_suspended:
            fallback_reason = "suspended_reference_reset"
        elif prior_suspended:
            fallback_reason = "resumption_complex_action"
        elif has_explicit_action or factor_changed[position]:
            fallback_reason = "complex_corporate_action"
        else:
            unresolved.append(
                {
                    "date": date_value,
                    "candidate": candidate,
                    "bundle_prev_close": float(reference[position]),
                    "absolute_difference": difference,
                }
            )
            continue
        output[position] = float(reference[position])
        methods[position] = f"bundle_reference:{fallback_reason}"
        fallback_examples.append(
            {
                "date": date_value,
                "reason": fallback_reason,
                "candidate_method": method,
                "candidate": candidate,
                "rqalpha_factor_preclose": float(factor_based[position]),
                "bundle_prev_close": float(reference[position]),
                "absolute_difference": difference,
            }
        )

    if unresolved:
        raise ValueError(
            "Unexplained preclose differences outside tolerance: "
            f"{unresolved[:10]}"
        )
    unique, counts = np.unique(methods, return_counts=True)
    audit: dict[str, object] = {
        "method_counts": {
            str(method): int(count) for method, count in zip(unique, counts, strict=True)
        },
        "controlled_fallback_count": len(fallback_examples),
        "controlled_fallback_examples": fallback_examples,
        "unexplained_difference_count": 0,
    }
    return output, factor_based, methods, audit


def preclose_crosscheck(
    reconstructed: np.ndarray,
    bundled: np.ndarray,
    *,
    absolute_tolerance: float = 0.011,
) -> dict[str, float | int]:
    """Validate reconstructed values against RQAlpha's bundled reference field."""

    left = np.asarray(reconstructed, dtype="float64")
    right = np.asarray(bundled, dtype="float64")
    if left.shape != right.shape:
        raise ValueError("Preclose cross-check arrays have different shapes.")
    mask = np.isfinite(left) & np.isfinite(right) & (right > 0)
    if not mask.any():
        raise ValueError("No valid observations for the preclose cross-check.")
    absolute = np.abs(left[mask] - right[mask])
    # RQAlpha's prev_close is normally rounded to the exchange quotation tick.
    unresolved = int((absolute > absolute_tolerance).sum())
    return {
        "observations": int(mask.sum()),
        "median_absolute_difference": float(np.median(absolute)),
        "maximum_absolute_difference": float(absolute.max()),
        "outside_tolerance": unresolved,
    }

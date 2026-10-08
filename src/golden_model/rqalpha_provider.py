"""RQAlpha monthly-bundle provider for the production replication path."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .corporate_actions import construct_audited_preclose, preclose_crosscheck
from .data_provider import MarketDataBundle, read_json, sha256_file, write_json
from .index_membership import rebuild_official_membership


REQUIRED_BUNDLE_FILES = {
    "stocks.h5",
    "indexes.h5",
    "suspended_days.h5",
    "trading_dates.npy",
    "dividends.h5",
    "split_factor.h5",
    "ex_cum_factor.h5",
    "instruments.pk",
    "share_transformation.json",
}
CACHE_SCHEMA = "rqalpha-official-membership-v3"


def _date_from_bundle(value: np.ndarray | np.integer | int) -> pd.Timestamp:
    integer = int(value)
    if integer > 100_000_000:
        integer //= 1_000_000
    return pd.Timestamp(str(integer))


def _provider_code(value: str) -> str:
    digits, exchange = value.split(".")
    prefix = "sh" if exchange == "XSHG" else "sz"
    return f"{prefix}.{digits}"


class RQAlphaProvider:
    def __init__(self, config: dict[str, Any], root: Path) -> None:
        self.config = config
        self.root = root.resolve()
        self.cache_root = (self.root / config["data"]["cache_dir"] / "rqalpha").resolve()
        self.market_cache = self.cache_root / "market"
        self.notice_cache = self.cache_root / "official_index_notices"

    def _bundle_candidates(self) -> list[Path]:
        configured = self.config["data"].get("rqalpha_bundle_dir", "data/cache/rqalpha/bundle")
        preferred = (self.root / configured).resolve()
        # The RQAlpha CLI appends "bundle" to -d.  The second candidate keeps
        # compatibility with a pre-existing download made with -d .../bundle.
        return [preferred, preferred / "bundle"]

    @staticmethod
    def _is_bundle(path: Path) -> bool:
        return path.is_dir() and REQUIRED_BUNDLE_FILES.issubset(
            {item.name for item in path.iterdir() if item.is_file()}
        )

    def _ensure_bundle(self, refresh: bool) -> Path:
        if not refresh:
            for candidate in self._bundle_candidates():
                if self._is_bundle(candidate):
                    return candidate
        destination = self.cache_root
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "rqalpha", "download-bundle", "-d", str(destination)],
            cwd=self.root,
            check=True,
        )
        bundle = destination / "bundle"
        if not self._is_bundle(bundle):
            raise ValueError(f"RQAlpha download did not create a valid bundle at {bundle}.")
        return bundle

    @staticmethod
    def _bundle_manifest(bundle: Path) -> dict[str, Any]:
        files: dict[str, Any] = {}
        for name in sorted(REQUIRED_BUNDLE_FILES):
            path = bundle / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        return {"required_files": files}

    @staticmethod
    def _read_instruments(bundle: Path) -> dict[str, dict[str, Any]]:
        # instruments.pk is an official RQAlpha bundle artifact.  Loading is
        # restricted to the freshly downloaded/validated bundle path.
        with (bundle / "instruments.pk").open("rb") as handle:
            raw = pickle.load(handle)  # noqa: S301 - trusted official bundle only
        records: dict[str, dict[str, Any]] = {}
        iterable = raw.values() if isinstance(raw, dict) else raw
        for value in iterable:
            if hasattr(value, "_dict"):
                item = dict(value._dict)
            elif isinstance(value, dict):
                item = dict(value)
            else:
                continue
            code = str(item.get("order_book_id", ""))
            if code:
                records[code] = item
        if not records:
            raise ValueError("RQAlpha instruments.pk contains no readable instruments.")
        return records

    @staticmethod
    def _calendar(bundle: Path) -> pd.DatetimeIndex:
        values = np.load(bundle / "trading_dates.npy", allow_pickle=False)
        dates = pd.to_datetime(values.astype(str), format="%Y%m%d", errors="raise")
        if not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("RQAlpha trading calendar is not strictly increasing.")
        return pd.DatetimeIndex(dates)

    @staticmethod
    def _factor_records(store: h5py.File, code: str) -> np.ndarray | None:
        return store[code][:] if code in store else None

    @staticmethod
    def _suspended_dates(store: h5py.File, code: str) -> set[int]:
        return set(map(int, store[code][:])) if code in store else set()

    def _stock_prices(
        self,
        bundle: Path,
        codes: list[str],
        instruments: dict[str, dict[str, Any]],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        pieces: list[pd.DataFrame] = []
        crosschecks: list[dict[str, float | int]] = []
        method_counts: dict[str, int] = {}
        controlled_fallbacks: list[dict[str, Any]] = []
        with (
            h5py.File(bundle / "stocks.h5", "r") as bars_store,
            h5py.File(bundle / "ex_cum_factor.h5", "r") as factor_store,
            h5py.File(bundle / "suspended_days.h5", "r") as suspended_store,
            h5py.File(bundle / "dividends.h5", "r") as dividend_store,
            h5py.File(bundle / "split_factor.h5", "r") as split_store,
        ):
            for code in codes:
                if code not in bars_store:
                    raise ValueError(f"Historical constituent absent from RQAlpha stocks.h5: {code}")
                metadata = instruments.get(code)
                if metadata is None:
                    raise ValueError(f"Historical constituent absent from instruments.pk: {code}")
                bars = bars_store[code][:]
                suspended = self._suspended_dates(suspended_store, code)
                reconstructed, factor_preclose, methods, audit = construct_audited_preclose(
                    bars,
                    self._factor_records(factor_store, code),
                    self._factor_records(dividend_store, code),
                    self._factor_records(split_store, code),
                    suspended,
                )
                dates = pd.to_datetime(
                    (bars["datetime"] // 1_000_000).astype(str), format="%Y%m%d"
                )
                mask = (dates >= start) & (dates <= end)
                if not mask.any():
                    raise ValueError(f"No RQAlpha price observations in range for {code}.")
                selected = bars[mask]
                selected_dates = dates[mask]
                preclose = reconstructed[mask]
                factor_selected = factor_preclose[mask]
                selected_methods = methods[mask]
                unique_methods, method_totals = np.unique(selected_methods, return_counts=True)
                for method, count in zip(unique_methods, method_totals, strict=True):
                    method_counts[str(method)] = method_counts.get(str(method), 0) + int(count)
                start_integer = int(start.strftime("%Y%m%d"))
                end_integer = int(end.strftime("%Y%m%d"))
                for example in audit["controlled_fallback_examples"]:
                    if start_integer <= int(example["date"]) <= end_integer:
                        controlled_fallbacks.append({"code": code, **example})
                check = preclose_crosscheck(factor_selected, selected["prev_close"])
                check["code"] = code
                crosschecks.append(check)
                date_ints = (selected["datetime"] // 1_000_000).astype("int64")
                is_suspended = np.fromiter(
                    (int(value) in suspended for value in date_ints),
                    dtype=bool,
                    count=len(date_ints),
                )
                listed_raw = str(metadata.get("listed_date", ""))[:10]
                delisted_raw = str(metadata.get("de_listed_date", ""))[:10]
                listed = pd.Timestamp(listed_raw)
                delisted = (
                    pd.Timestamp("2262-01-01")
                    if delisted_raw in {"", "0000-00-00", "2999-12-31"}
                    else pd.Timestamp(delisted_raw)
                )
                status = np.where(is_suspended, "suspended", "trading").astype(object)
                status[selected_dates < listed] = "pre_listing"
                status[selected_dates >= delisted] = "post_delisting"
                frame = pd.DataFrame(
                    {
                        "date": selected_dates,
                        "code": _provider_code(code),
                        "rqalpha_code": code,
                        "open": selected["open"].astype(float),
                        "close": selected["close"].astype(float),
                        "preclose": preclose,
                        "rqalpha_factor_preclose": factor_selected,
                        "bundle_prev_close": selected["prev_close"].astype(float),
                        "turn": np.nan,
                        "tradestatus": (~is_suspended).astype(int),
                        "security_status": status,
                    }
                )
                pieces.append(frame)
        output = pd.concat(pieces, ignore_index=True)
        outside = sum(int(item["outside_tolerance"]) for item in crosschecks)
        validation = {
            "method": (
                "ordinary prior raw close; explicit cash/split formula; controlled bundle "
                "reference only for classified complex actions/suspension resets"
            ),
            "observations": sum(int(item["observations"]) for item in crosschecks),
            "rqalpha_factor_vs_bundle_outside_one_tick": outside,
            "rqalpha_factor_vs_bundle_maximum_absolute_difference": max(
                float(item["maximum_absolute_difference"]) for item in crosschecks
            ),
            "method_counts": method_counts,
            "controlled_fallback_count": len(controlled_fallbacks),
            "controlled_fallback_examples": sorted(
                controlled_fallbacks,
                key=lambda item: float(item["absolute_difference"]),
                reverse=True,
            )[:100],
            "unexplained_difference_count": 0,
        }
        return output, validation

    @staticmethod
    def _index_prices(
        bundle: Path, execution_assets: dict[str, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        with h5py.File(bundle / "indexes.h5", "r") as store:
            for provider_code in execution_assets.values():
                digits = provider_code.split(".")[-1]
                code = f"{digits}.XSHG"
                if code not in store:
                    raise ValueError(f"Execution index absent from RQAlpha bundle: {code}")
                bars = store[code][:]
                dates = pd.to_datetime(
                    (bars["datetime"] // 1_000_000).astype(str), format="%Y%m%d"
                )
                mask = (dates >= start) & (dates <= end)
                selected = bars[mask]
                pieces.append(
                    pd.DataFrame(
                        {
                            "date": dates[mask],
                            "code": provider_code,
                            "open": selected["open"].astype(float),
                            "close": selected["close"].astype(float),
                            "preclose": selected["prev_close"].astype(float),
                            "turn": np.nan,
                            "tradestatus": 1,
                            "security_status": "trading",
                        }
                    )
                )
        return pd.concat(pieces, ignore_index=True)

    def _cache_paths(self) -> dict[str, Path]:
        return {
            "trade_dates": self.market_cache / "trade_dates.csv.gz",
            "constituents": self.market_cache / "constituents.csv.gz",
            "stock_prices": self.market_cache / "stock_prices.csv.gz",
            "index_prices": self.market_cache / "index_prices.csv.gz",
            "provenance": self.market_cache / "provenance.json",
        }

    def _read_cache(self) -> MarketDataBundle | None:
        paths = self._cache_paths()
        if not all(path.is_file() for path in paths.values()):
            return None
        provenance = read_json(paths["provenance"])
        if provenance.get("cache_schema") != CACHE_SCHEMA:
            return None
        frames: dict[str, pd.DataFrame] = {}
        for name in ("trade_dates", "constituents", "stock_prices", "index_prices"):
            expected = provenance["cache_files"][paths[name].name]["sha256"]
            if sha256_file(paths[name]) != expected:
                raise ValueError(f"RQAlpha converted cache checksum failed: {paths[name].name}")
            frame = pd.read_csv(paths[name])
            for column in ("date", "snapshot_date", "provider_update_date"):
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column], errors="raise")
            frames[name] = frame
        return MarketDataBundle(provenance=provenance, **frames)

    def _write_cache(self, bundle: MarketDataBundle) -> None:
        self.market_cache.mkdir(parents=True, exist_ok=True)
        paths = self._cache_paths()
        for name in ("trade_dates", "constituents", "stock_prices", "index_prices"):
            temporary = paths[name].with_suffix(paths[name].suffix + ".tmp")
            getattr(bundle, name).to_csv(temporary, index=False, compression="gzip")
            temporary.replace(paths[name])
        provenance = deepcopy(bundle.provenance)
        provenance["cache_files"] = {
            paths[name].name: {
                "bytes": paths[name].stat().st_size,
                "sha256": sha256_file(paths[name]),
            }
            for name in ("trade_dates", "constituents", "stock_prices", "index_prices")
        }
        write_json(paths["provenance"], provenance)

    def load(self, refresh: bool = False) -> MarketDataBundle:
        if not refresh:
            cached = self._read_cache()
            if cached is not None:
                return cached
        bundle_path = self._ensure_bundle(refresh)
        calendar = self._calendar(bundle_path)
        requested_start = pd.Timestamp(self.config["data"]["start_date"])
        requested_end = pd.Timestamp(self.config["data"]["end_date"])
        warmup = requested_start - pd.Timedelta(
            days=int(self.config["data"].get("warmup_calendar_days", 14))
        )
        index_probe = self._index_prices(
            bundle_path,
            self.config["data"]["execution_assets"],
            warmup,
            requested_end,
        )
        actual_end = min(requested_end, index_probe["date"].max())
        usable_calendar = calendar[(calendar >= warmup) & (calendar <= actual_end)]
        if actual_end < requested_end:
            cutoff_reason = "RQAlpha bundle ends before configured requested end"
        else:
            cutoff_reason = "configured requested end available"
        instruments = self._read_instruments(bundle_path)
        transformation_path = bundle_path / "share_transformation.json"
        share_transformations = json.loads(transformation_path.read_text(encoding="utf-8"))
        constituents, events, membership_manifest = rebuild_official_membership(
            self.config,
            self.notice_cache,
            calendar,
            instruments,
            share_transformations,
            {
                "source_file": "share_transformation.json",
                "source_sha256": sha256_file(transformation_path),
            },
        )
        rq_codes = sorted(
            {
                f"{code.split('.')[1]}.{'XSHG' if code.startswith('sh.') else 'XSHE'}"
                for code in constituents["code"].astype(str)
            }
        )
        stock_prices, preclose_validation = self._stock_prices(
            bundle_path, rq_codes, instruments, warmup, actual_end
        )
        index_prices = index_probe.loc[index_probe["date"] <= actual_end].copy()
        trade_dates = pd.DataFrame({"date": usable_calendar, "is_trading_day": 1})

        # Membership codes are normalized to the provider-neutral sh./sz. form.
        constituents = constituents.loc[constituents["snapshot_date"] <= actual_end].copy()
        bundle_manifest = self._bundle_manifest(bundle_path)
        provenance: dict[str, Any] = {
            "cache_schema": CACHE_SCHEMA,
            "provider": "RQAlpha free monthly bundle",
            "provider_version": importlib.metadata.version("rqalpha"),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_path": str(bundle_path.relative_to(self.root)),
            "bundle": bundle_manifest,
            "requested_start": requested_start.date().isoformat(),
            "requested_end": requested_end.date().isoformat(),
            "actual_end": actual_end.date().isoformat(),
            "cutoff_reason": cutoff_reason,
            "historical_membership": membership_manifest,
            "preclose_validation": preclose_validation,
            "suspension_source": "suspended_days.h5",
            "turnover_field": "unavailable in free bundle; unused by strategy",
            "baostock_role": "migration validation only; excluded from production frames",
            "production_data": True,
            "synthetic_fallback": False,
        }
        result = MarketDataBundle(
            trade_dates=trade_dates,
            constituents=constituents,
            stock_prices=stock_prices,
            index_prices=index_prices,
            provenance=provenance,
        )
        self._write_cache(result)
        return self._read_cache() or result

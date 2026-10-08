"""Fail-closed point-in-time index membership from official CSI/SSE notices."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
from html import unescape
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from lxml import html as lxml_html

from .data_provider import sha256_file, write_json


PARSER_VERSION = "official-index-events-v2"
SEARCH_URL = "https://www.csindex.com.cn/csindex-home/search/search-content"
DETAIL_URL = "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById"
DETAIL_PAGE = "https://www.csindex.com.cn/#/about/newsDetail?id={notice_id}"
ANCHOR_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/"
    "file/autofile/cons/{index_code}cons.xls"
)
SSE_LIST_URL = "https://www.sse.com.cn/market/sseindex/diclosure/{page}"
SSE_ORIGIN = "https://www.sse.com.cn"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36 "
        "Edg/91.0.864.48"
    )
}


@dataclass(frozen=True)
class MembershipEvent:
    index_code: str
    announcement_date: str
    effective_date: str
    add_list: list[str]
    remove_list: list[str]
    source_url: str
    source_sha256: str
    parser_version: str = PARSER_VERSION
    evidence: list[dict[str, Any]] = field(default_factory=list)


def normalize_order_book_id(value: Any) -> str:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        raise ValueError(f"Invalid security code: {value!r}")
    code = f"{int(digits):06d}"
    exchange = "XSHG" if code.startswith(("5", "6", "9")) else "XSHE"
    return f"{code}.{exchange}"


def provider_code(value: str) -> str:
    """Convert an RQAlpha identifier to the project's provider-neutral form."""

    digits, exchange = value.split(".")
    return f"{'sh' if exchange == 'XSHG' else 'sz'}.{digits}"


class OfficialNoticeClient:
    """Single-session, rate-limited official-source client with durable cache."""

    def __init__(self, cache_dir: Path, timeout: int = 60) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last_request = 0.0

    def _get(self, url: str, *, params: dict[str, Any] | None = None) -> requests.Response:
        error: Exception | None = None
        for attempt in range(5):
            delay = 3.0 - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                self._last_request = time.monotonic()
                if response.status_code == 200 and response.content:
                    return response
                if response.status_code in {403, 404}:
                    raise PermissionError(
                        f"Official source returned HTTP {response.status_code}: {response.url}"
                    )
                raise RuntimeError(
                    f"Official source returned HTTP {response.status_code}: {response.url}"
                )
            except PermissionError:
                raise
            except (requests.RequestException, RuntimeError) as exc:
                error = exc
                time.sleep((5, 10, 20, 30, 45)[attempt])
        raise RuntimeError(f"Official source request failed: {url}") from error

    def search(self, query: str) -> list[dict[str, Any]]:
        key = sha256(query.encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / f"search-{key}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            params = {
                "lang": "cn",
                "searchInput": query,
                "pageNum": 1,
                "pageSize": 100,
                "sortField": "date",
                "dateRange": "all",
                "contentType": "announcement",
            }
            payload = self._get(SEARCH_URL, params=params).json()
            write_json(path, payload)
        if payload.get("code") != "200" or not isinstance(payload.get("data"), list):
            raise ValueError(f"Malformed official search response for {query!r}.")
        return payload["data"]

    def detail(self, notice_id: int) -> dict[str, Any]:
        path = self.cache_dir / f"notice-{notice_id}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = self._get(DETAIL_URL, params={"id": notice_id}).json()
            write_json(path, payload)
        data = payload.get("data")
        if not isinstance(data, dict) or int(data.get("id", -1)) != notice_id:
            raise ValueError(f"Malformed official notice response: {notice_id}.")
        return data

    def download(self, url: str, label: str) -> Path:
        suffix = Path(url.split("?", 1)[0]).suffix.lower() or ".bin"
        path = self.cache_dir / f"{label}{suffix}"
        if not path.is_file():
            content = self._get(url).content
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        if path.stat().st_size < 100:
            raise ValueError(f"Official attachment is unexpectedly small: {url}")
        return path

    def text(self, url: str, label: str) -> tuple[str, str]:
        path = self.cache_dir / f"{label}.html"
        if not path.is_file():
            content = self._get(url).content
            temporary = path.with_suffix(".html.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        raw = path.read_bytes()
        # Both official sites declare UTF-8, but older pages occasionally omit
        # a useful HTTP charset.  UTF-8 with replacement preserves code tables.
        return raw.decode("utf-8", errors="replace"), sha256(raw).hexdigest()


def _clean_html(value: str) -> str:
    return unescape(re.sub(r"<[^>]+>", " ", value)).replace("\xa0", " ")


def _next_session(value: pd.Timestamp, trading_dates: pd.DatetimeIndex) -> pd.Timestamp:
    position = trading_dates.searchsorted(value, side="right")
    if position >= len(trading_dates):
        raise ValueError(f"No trading session after {value.date()}.")
    return trading_dates[position]


def _effective_date(detail: dict[str, Any], trading_dates: pd.DatetimeIndex) -> pd.Timestamp:
    text = _clean_html(str(detail.get("content", "")))
    publish = pd.Timestamp(detail["publishDate"])
    observations: list[tuple[pd.Timestamp, bool]] = []
    occupied: list[tuple[int, int]] = []
    full_pattern = re.compile(r"(20\s*\d\s*\d\s*\d)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
    for match in full_pattern.finditer(text):
        year = int(re.sub(r"\s+", "", match.group(1)))
        candidate = pd.Timestamp(year=year, month=int(match.group(2)), day=int(match.group(3)))
        suffix = text[match.end() : match.end() + 40]
        after_close = bool(re.search(r"(?:收盘|收市|盘)\s*后", suffix))
        observations.append((candidate, after_close))
        occupied.append(match.span())

    partial_pattern = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
    for match in partial_pattern.finditer(text):
        if any(start <= match.start() < stop for start, stop in occupied):
            continue
        month, day = match.groups()
        candidate = pd.Timestamp(year=publish.year, month=int(month), day=int(day))
        if candidate < publish - pd.Timedelta(days=5):
            candidate = candidate.replace(year=publish.year + 1)
        suffix = text[match.end() : match.end() + 40]
        after_close = bool(re.search(r"(?:收盘|收市|盘)\s*后", suffix))
        observations.append((candidate, after_close))

    english_pattern = re.compile(
        r"(?:close\s+of\s+)?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\.?\s+(\d{1,2}),\s*(20\d{2})",
        re.I,
    )
    months = {name.lower(): number for number, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
        start=1,
    )}
    for match in english_pattern.finditer(text):
        candidate = pd.Timestamp(
            year=int(match.group(3)), month=months[match.group(1).lower()], day=int(match.group(2))
        )
        context = text[max(0, match.start() - 30) : match.end() + 30]
        observations.append((candidate, bool(re.search(r"close\s+of", context, re.I))))

    candidates = sorted(
        {(value.normalize(), after_close) for value, after_close in observations if value >= publish},
        key=lambda item: item[0],
    )
    later = [item for item in candidates if item[0] > publish]
    if later:
        effective, after_close = later[0]
    elif candidates:
        effective, after_close = candidates[0]
    else:
        raise ValueError(f"Notice {detail.get('id')} has no explicit effective date.")
    if after_close:
        effective = _next_session(effective, trading_dates)
    return effective.normalize()


def _resolve_symbol(symbol: Any, symbol_codes: dict[str, list[str]]) -> str | None:
    cleaned = re.sub(r"\s+", "", str(symbol)).replace("-", "")
    if not cleaned:
        return None
    matches = symbol_codes.get(cleaned, [])
    if len(matches) != 1:
        raise ValueError(f"Official constituent name {cleaned!r} does not map uniquely: {matches}.")
    return matches[0]


def _codes_from_excel(
    path: Path,
    index_code: str,
    index_name: str,
    symbol_codes: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    sheets = pd.read_excel(path, sheet_name=None)
    if "调入" in sheets and "调出" in sheets:
        output: dict[str, list[str]] = {}
        for direction in ("调入", "调出"):
            frame = sheets[direction]
            code_col = next((c for c in frame.columns if "指数代码" in str(c)), None)
            security_col = next(
                (
                    c
                    for c in frame.columns
                    if any(token in str(c) for token in ("证券代码", "股票代码"))
                ),
                None,
            )
            if code_col is None or security_col is None:
                raise ValueError(f"Official workbook {path.name} has unexpected columns.")
            index_values = frame[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            chosen = frame.loc[index_values.eq(index_code), security_col].dropna()
            output[direction] = [normalize_order_book_id(v) for v in chosen]
        return output["调入"], output["调出"]

    if "Addition" in sheets and "Deletion" in sheets:
        output = {}
        for sheet_name, direction in (("Addition", "调入"), ("Deletion", "调出")):
            frame = sheets[sheet_name]
            code_col = next((c for c in frame.columns if "Index Code" in str(c)), None)
            security_col = next((c for c in frame.columns if "Security Code" in str(c)), None)
            if code_col is None or security_col is None:
                raise ValueError(f"Official workbook {path.name} has unexpected English columns.")
            index_values = frame[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            chosen = frame.loc[index_values.eq(index_code), security_col].dropna()
            output[direction] = [normalize_order_book_id(v) for v in chosen]
        return output["调入"], output["调出"]

    for frame in sheets.values():
        index_column = next((c for c in frame.columns if "指数代码" in str(c)), None)
        if index_column is None:
            continue
        index_values = (
            frame[index_column].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        )
        rows = frame.loc[index_values.eq(index_code)]
        remove_column = next((c for c in frame.columns if str(c).strip() == "调出"), None)
        add_column = next((c for c in frame.columns if str(c).strip() == "调入"), None)
        if not rows.empty and remove_column is not None and add_column is not None:
            remove = [normalize_order_book_id(value) for value in rows[remove_column] if str(value) != "-"]
            add = [normalize_order_book_id(value) for value in rows[add_column] if str(value) != "-"]
            if len(add) == len(remove):
                return add, remove

    # Some official workbooks use merged two-level headings, and may list a
    # successor name before its exchange code exists.  Parse the raw row and
    # resolve such a name only when RQAlpha metadata has exactly one match.
    raw_sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    for frame in raw_sheets.values():
        if frame.empty or frame.shape[1] < 6:
            continue
        for row_index, row in frame.iterrows():
            first = re.sub(r"\.0$", "", str(row.iloc[0])).zfill(6)
            if first != index_code:
                continue
            # The merged direction header normally appears only once at the
            # top of a long workbook, so search all preceding rows.
            header_rows = frame.iloc[:row_index]
            remove_start: int | None = None
            add_start: int | None = None
            for _, header in header_rows.iterrows():
                for column, value in enumerate(header.astype(str)):
                    if "调出" in value and remove_start is None:
                        remove_start = column
                    if "调入" in value and add_start is None:
                        add_start = column
            if remove_start is None or add_start is None or remove_start >= add_start:
                continue

            def parse_pairs(values: list[Any]) -> list[str]:
                parsed: list[str] = []
                for offset in range(0, len(values) - 1, 2):
                    code_value, name_value = values[offset], values[offset + 1]
                    code_text = re.sub(r"\.0$", "", str(code_value)).strip()
                    if re.fullmatch(r"[036]\d{5}", code_text):
                        parsed.append(normalize_order_book_id(code_text))
                    elif code_text in {"", "-", "nan", "None"}:
                        resolved = _resolve_symbol(name_value, symbol_codes)
                        if resolved is not None:
                            parsed.append(resolved)
                return parsed

            remove = parse_pairs(row.iloc[remove_start:add_start].tolist())
            add = parse_pairs(row.iloc[add_start:].tolist())
            if remove or add:
                return add, remove

    # Newer temporary-adjustment workbooks use one sheet per affected index.
    for sheet_name, frame in sheets.items():
        if index_name not in str(sheet_name) and index_code not in str(sheet_name):
            continue
        raw = frame.astype(str)
        rows = [" ".join(row) for row in raw.to_numpy().tolist()]
        pairs = [re.findall(r"(?<!\d)([036]\d{5})(?!\d)", row) for row in rows]
        pairs = [values for values in pairs if len(values) >= 2]
        if pairs:
            remove = [normalize_order_book_id(values[0]) for values in pairs]
            add = [normalize_order_book_id(values[1]) for values in pairs]
            return add, remove
    raise ValueError(f"Could not parse {index_name} from official workbook {path.name}.")


def _codes_from_pdf(path: Path, index_name: str) -> tuple[list[str], list[str]]:
    from pypdf import PdfReader

    lines: list[str] = []
    for page in PdfReader(str(path)).pages:
        value = page.extract_text(extraction_mode="layout") or ""
        lines.extend(value.splitlines())
    normalized_name = re.sub(r"\s+", "", index_name)
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if normalized_name in re.sub(r"\s+", "", line)
            and "调整名单" in re.sub(r"\s+", "", line)
        ),
        None,
    )
    if start is None:
        raise ValueError(f"Could not find {index_name} table in {path.name}.")
    selected: list[str] = []
    for line in lines[start + 1 :]:
        normalized_line = re.sub(r"\s+", "", line)
        if selected and "指数样本调整名单" in normalized_line and normalized_name not in normalized_line:
            break
        selected.append(line)
    pairs: list[list[str]] = []
    for line in selected:
        values = re.findall(r"(?<!\d)([036]\d{5})(?!\d)", line)
        if len(values) == 2:
            pairs.append(values)
    if not pairs:
        raise ValueError(f"No paired security codes found for {index_name} in {path.name}.")
    remove = [normalize_order_book_id(values[0]) for values in pairs]
    add = [normalize_order_book_id(values[1]) for values in pairs]
    return add, remove


def _codes_from_html(
    content: str,
    index_name: str,
    index_code: str | None = None,
) -> tuple[list[str], list[str]]:
    # Older official notices sometimes insert ordinary whitespace between
    # Chinese text and digits (for example ``中证 500``).  Match the heading
    # in the original HTML so whitespace inside tags remains intact.
    heading_pattern = r"\s*".join(re.escape(character) for character in index_name)
    heading = re.search(heading_pattern, content)
    if heading is None:
        raise ValueError(f"Notice HTML does not mention {index_name}.")
    scoped_content = content[heading.start() :]
    table_matches = list(re.finditer(r"<table\b.*?</table>", scoped_content, re.I | re.S))
    if not table_matches:
        raise ValueError(f"Notice HTML has no table after {index_name} heading.")
    tables: list[tuple[pd.DataFrame, bool]] = []
    previous_end = 0
    normalized_name = re.sub(r"\s+", "", index_name)
    for table in table_matches:
        section_html = scoped_content[previous_end : table.start()]
        section_text = re.sub(r"<[^>]+>", " ", section_html)
        # Introductory paragraphs often name every affected index.  Only the
        # nearest heading immediately before a four-column table identifies
        # that table, so do not treat an earlier mention as table context.
        normalized_section = re.sub(r"\s+", "", section_text)
        heading_end = normalized_section.rfind("调整名单")
        nearest_heading = (
            normalized_section[max(0, heading_end - 40) : heading_end + 4]
            if heading_end >= 0
            else normalized_section[-80:]
        )
        is_target_context = normalized_name in nearest_heading
        try:
            tables.extend(
                (frame, is_target_context)
                for frame in pd.read_html(StringIO(table.group(0)))
            )
        except ValueError:
            pass
        previous_end = table.end()

    pairs: list[tuple[str, str]] = []
    for frame, is_target_context in tables:
        if frame.shape[1] < 4:
            continue
        for values in frame.astype(str).to_numpy().tolist():
            def parsed_code(value: str) -> str | None:
                match = re.fullmatch(r"\s*(\d{1,6})(?:\.0)?\s*", value)
                if match is None:
                    return None
                code = match.group(1).zfill(6)
                return code if code.startswith(("0", "3", "6")) else None

            # Some official pages put the index code/name in the first two
            # columns, while simpler tables start directly with the removed
            # security.  Filtering by index_code lets us safely aggregate
            # multiple temporary adjustments published on the same page.
            offset = 0
            if index_code is not None and parsed_code(values[0]) == index_code:
                offset = 2
            elif index_code is not None:
                if len(values) >= 6 or not is_target_context:
                    continue
            if len(values) < offset + 4:
                continue
            left = parsed_code(values[offset])
            right = parsed_code(values[offset + 2])
            if left and right:
                pairs.append((left, right))
    # Preserve the notice order but remove exact duplicate table rows.
    pairs = list(dict.fromkeys(pairs))
    remove = [normalize_order_book_id(left) for left, _ in pairs]
    add = [normalize_order_book_id(right) for _, right in pairs]
    if remove and len(remove) == len(add):
        return add, remove
    raise ValueError(f"Could not parse {index_name} from notice HTML.")


def _parse_event_codes(
    client: OfficialNoticeClient,
    detail: dict[str, Any],
    index_code: str,
    index_name: str,
    symbol_codes: dict[str, list[str]],
) -> tuple[list[str], list[str], str, str]:
    failures: list[str] = []
    candidates: list[tuple[list[str], list[str], str, str]] = []
    attachment_urls = [
        str(attachment.get("fileUrl", ""))
        for attachment in detail.get("enclosureList", [])
    ]
    attachment_urls.extend(
        re.findall(r'href=["\']([^"\']+\.(?:xlsx?|pdf))', str(detail.get("content", "")), re.I)
    )
    source_base = str(detail.get("_source_url", "https://www.csindex.com.cn/"))
    for raw_url in dict.fromkeys(attachment_urls):
        url = urljoin(source_base, raw_url)
        if not url:
            continue
        try:
            attachment_key = sha256(url.encode("utf-8")).hexdigest()[:12]
            path = client.download(
                url, f"notice-{detail['id']}-attachment-{attachment_key}"
            )
            if path.suffix.lower() in {".xls", ".xlsx"}:
                add, remove = _codes_from_excel(path, index_code, index_name, symbol_codes)
            elif path.suffix.lower() == ".pdf":
                add, remove = _codes_from_pdf(path, index_name)
            else:
                continue
            candidates.append((add, remove, url, sha256_file(path)))
        except (KeyError, ValueError, RuntimeError, PermissionError) as exc:
            failures.append(str(exc))
    try:
        add, remove = _codes_from_html(
            str(detail.get("content", "")), index_name, index_code
        )
        content = str(detail.get("content", "")).encode("utf-8")
        candidates.append(
            (
                add,
                remove,
                str(detail.get("_source_url", DETAIL_PAGE.format(notice_id=detail["id"]))),
                str(detail.get("_source_sha256", sha256(content).hexdigest())),
            )
        )
    except ValueError as exc:
        failures.append(str(exc))
    if candidates:
        # A single official notice may carry several attachments/tables for
        # independent temporary changes effective on the same date.  The
        # complete official representation is the candidate with the largest
        # balanced change set (normally the HTML page that contains all rows).
        return max(candidates, key=lambda item: (len(item[0]), len(item[1])))
    raise ValueError(
        f"Notice {detail.get('id')} cannot be parsed for {index_name}: {'; '.join(failures)}"
    )


def _ranked_reserves_from_excel(path: Path, index_code: str) -> list[str]:
    """Read an official ranked reserve list from a regular-review workbook."""

    workbook = pd.ExcelFile(path)
    candidates: list[list[tuple[int, str]]] = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
        if frame.shape[1] < 5:
            continue
        code_column = frame.iloc[:, 0].fillna("").str.replace(r"\.0$", "", regex=True)
        selected = frame.loc[code_column.str.zfill(6).eq(index_code)]
        ranked: list[tuple[int, str]] = []
        for values in selected.iloc[:, :5].fillna("").astype(str).to_numpy().tolist():
            rank_match = re.fullmatch(r"\s*(\d+)(?:\.0)?\s*", values[2])
            code_match = re.fullmatch(r"\s*([036]\d{5})(?:\.0)?\s*", values[3])
            if rank_match and code_match:
                ranked.append(
                    (int(rank_match.group(1)), normalize_order_book_id(code_match.group(1)))
                )
        if ranked:
            candidates.append(ranked)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one ranked reserve list for {index_code} in {path.name}, "
            f"found {len(candidates)}."
        )
    ranked = sorted(candidates[0])
    ranks = [rank for rank, _ in ranked]
    if ranks != list(range(1, len(ranked) + 1)):
        raise ValueError(f"Non-contiguous reserve ranks for {index_code} in {path.name}.")
    codes = [code for _, code in ranked]
    if len(codes) != len(set(codes)):
        raise ValueError(f"Duplicate reserves for {index_code} in {path.name}.")
    return codes


def _notice_excel_attachments(
    client: OfficialNoticeClient, detail: dict[str, Any]
) -> list[tuple[str, Path]]:
    urls = [
        str(attachment.get("fileUrl", ""))
        for attachment in detail.get("enclosureList", [])
    ]
    urls.extend(
        re.findall(r'href=["\']([^"\']+\.xlsx?)(?:\?[^"\']*)?["\']', str(detail.get("content", "")), re.I)
    )
    source_base = str(detail.get("_source_url", DETAIL_PAGE.format(notice_id=detail["id"])))
    output: list[tuple[str, Path]] = []
    for raw_url in dict.fromkeys(urls):
        url = urljoin(source_base, raw_url)
        if url:
            output.append((url, client.download(url, f"notice-{detail['id']}-attachment")))
    return output


def _derive_official_cascade_event(
    client: OfficialNoticeClient,
    specification: dict[str, Any],
    details_by_id: dict[int, dict[str, Any]],
    parsed_events: dict[pd.Timestamp, MembershipEvent],
    index_code: str,
) -> tuple[pd.Timestamp, MembershipEvent]:
    """Derive a cross-index cascade solely from hashed official evidence.

    A deletion from a larger-cap index promotes its first published reserve.
    If that reserve is a CSI 500 member, the CSI 500 vacancy is filled by the
    first still-unused security in its own published reserve list.  Expected
    codes in configuration are assertions, never the source of the result.
    """

    effective = pd.Timestamp(specification["effective_date"])
    reserve_effective = pd.Timestamp(specification["reserve_effective_date"])
    trigger_id = int(specification["trigger_notice_id"])
    reserve_id = int(specification["reserve_notice_id"])
    try:
        trigger = details_by_id[trigger_id]
        reserve_notice = details_by_id[reserve_id]
    except KeyError as exc:
        raise ValueError(f"Missing official cascade evidence notice {exc.args[0]}.") from exc

    trigger_security = normalize_order_book_id(specification["trigger_security"])
    trigger_text = _clean_html(str(trigger.get("content", "")))
    if trigger_security.split(".", 1)[0] not in trigger_text or "退市" not in trigger_text:
        raise ValueError(
            f"Official notice {trigger_id} does not verify the configured delisting trigger."
        )

    reserve_files = _notice_excel_attachments(client, reserve_notice)
    if len(reserve_files) != 1:
        raise ValueError(
            f"Official reserve notice {reserve_id} has {len(reserve_files)} Excel attachments."
        )
    reserve_url, reserve_path = reserve_files[0]
    promoted = _ranked_reserves_from_excel(
        reserve_path, str(specification["higher_index_code"])
    )[0]
    lower_reserves = _ranked_reserves_from_excel(reserve_path, index_code)

    consumed = {
        code
        for event_date, event in parsed_events.items()
        if reserve_effective < event_date < effective
        for code in event.add_list
    }
    replacement = next(
        (code for code in lower_reserves if code not in consumed and code != promoted), None
    )
    if replacement is None:
        raise ValueError(f"No unused official {index_code} reserve for {effective.date()}.")

    expected_remove = normalize_order_book_id(specification["expected_remove"])
    expected_add = normalize_order_book_id(specification["expected_add"])
    if promoted != expected_remove or replacement != expected_add:
        raise ValueError(
            f"Official cascade assertion failed at {effective.date()}: "
            f"derived remove={promoted}, add={replacement}."
        )

    trigger_content = str(trigger.get("content", "")).encode("utf-8")
    trigger_url = str(
        trigger.get("_source_url", DETAIL_PAGE.format(notice_id=trigger_id))
    )
    event = MembershipEvent(
        index_code=index_code,
        announcement_date=str(trigger["publishDate"]),
        effective_date=effective.date().isoformat(),
        add_list=[replacement],
        remove_list=[promoted],
        source_url=trigger_url,
        source_sha256=str(
            trigger.get("_source_sha256", sha256(trigger_content).hexdigest())
        ),
        evidence=[
            {
                "role": "delisting_trigger",
                "notice_id": trigger_id,
                "source_url": trigger_url,
                "source_sha256": str(
                    trigger.get("_source_sha256", sha256(trigger_content).hexdigest())
                ),
                "security": trigger_security,
            },
            {
                "role": "ranked_reserve_lists",
                "notice_id": reserve_id,
                "source_url": reserve_url,
                "source_sha256": sha256_file(reserve_path),
                "higher_index_first_reserve": promoted,
                "lower_index_consumed_reserves": sorted(consumed.intersection(lower_reserves)),
                "lower_index_selected_reserve": replacement,
            },
        ],
    )
    return effective, event


def _official_anchor(
    client: OfficialNoticeClient, index_code: str, expected_count: int
) -> tuple[pd.Timestamp, set[str], dict[str, Any]]:
    url = ANCHOR_URL.format(index_code=index_code)
    path = client.download(url, f"anchor-{index_code}")
    frame = pd.read_excel(path)
    if len(frame) != expected_count or frame.shape[1] < 5:
        raise ValueError(
            f"Official {index_code} anchor count is {len(frame)}, expected {expected_count}."
        )
    as_of = pd.to_datetime(frame.iloc[:, 0].astype(str), format="%Y%m%d", errors="raise")
    if as_of.nunique() != 1:
        raise ValueError(f"Official {index_code} anchor contains multiple dates.")
    members = {normalize_order_book_id(value) for value in frame.iloc[:, 4]}
    if len(members) != expected_count:
        raise ValueError(f"Official {index_code} anchor has duplicate securities.")
    return as_of.iloc[0].normalize(), members, {
        "source_url": url,
        "source_sha256": sha256_file(path),
        "as_of_date": as_of.iloc[0].date().isoformat(),
        "member_count": len(members),
    }


def _discover_details(
    client: OfficialNoticeClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_dates: set[pd.Timestamp],
    required_notice_ids: Iterable[int] = (),
) -> list[dict[str, Any]]:
    queries = [
        "中证500",
        "调整中证500",
        "沪深300等指数样本",
        "关于调整沪深300和中证香港100等指数样本",
        "中证500指数样本临时调整",
        "上证50指数样本临时调整",
        "关于沪深300和中证香港100等指数定期调整结果",
        "Adjustment List for CSI 300 and CSI HK 100 Index etc.",
    ]
    items: dict[int, dict[str, Any]] = {}
    for query in queries:
        for item in client.search(query):
            published = pd.Timestamp(item["itemDate"])
            # Announcements precede their implementation.  Filtering the
            # search index before requesting details avoids unrelated archival
            # IDs and is deterministic against the configured event calendar.
            relevant = any(
                published <= effective <= published + pd.Timedelta(days=60)
                for effective in expected_dates
            )
            if start - pd.Timedelta(days=60) <= published <= end and relevant:
                items[int(item["id"])] = item
    details: list[dict[str, Any]] = []
    unavailable_path = client.cache_dir / "unavailable-notice-ids.json"
    unavailable: list[int] = []
    for notice_id in required_notice_ids:
        items.setdefault(int(notice_id), {"id": int(notice_id), "itemDate": str(start.date())})
    for notice_id in sorted(items):
        try:
            details.append(client.detail(notice_id))
        except (RuntimeError, PermissionError):
            unavailable.append(notice_id)
    if unavailable:
        # Missing details are surfaced later as missing effective events.  This
        # lets one archived ID fail without preventing other official sources
        # from being cached during the same run.
        unavailable_path.write_text(
            json.dumps(unavailable, indent=2), encoding="utf-8"
        )
    return sorted(details, key=lambda value: value["publishDate"])


def _discover_sse_details(
    client: OfficialNoticeClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_dates: set[pd.Timestamp],
    required_urls: set[str],
) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, str]] = {}
    for page_number in range(1, 38):
        page = "" if page_number == 1 else f"s_list_{page_number}.shtml"
        url = SSE_LIST_URL.format(page=page)
        content, _ = client.text(url, f"sse-list-{page_number}")
        tree = lxml_html.fromstring(content)
        oldest: pd.Timestamp | None = None
        for node in tree.xpath("//dd"):
            text = " ".join(node.text_content().split())
            date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
            links = node.xpath(".//a[@href]")
            if date_match is None or not links:
                continue
            published = pd.Timestamp(date_match.group(1))
            oldest = published if oldest is None else min(oldest, published)
            title = " ".join(links[0].text_content().split())
            href = str(links[0].get("href"))
            source_url = href if href.startswith("http") else SSE_ORIGIN + href
            if "上证50" not in title and source_url not in required_urls:
                continue
            relevant = any(
                published <= effective <= published + pd.Timedelta(days=60)
                for effective in expected_dates
            )
            if not relevant:
                continue
            entries[source_url] = {
                "publishDate": published.date().isoformat(),
                "title": title,
            }
        if oldest is not None and oldest < start - pd.Timedelta(days=60):
            break
    details: list[dict[str, Any]] = []
    for sequence, (url, entry) in enumerate(sorted(entries.items()), start=1):
        content, digest = client.text(url, f"sse-notice-{sequence}-{entry['publishDate']}")
        details.append(
            {
                "id": f"sse-{sha256(url.encode('utf-8')).hexdigest()[:12]}",
                "publishDate": entry["publishDate"],
                "title": entry["title"],
                "content": content,
                "enclosureList": [],
                "_source_url": url,
                "_source_sha256": digest,
            }
        )
    return sorted(details, key=lambda value: value["publishDate"])


def rebuild_official_membership(
    config: dict[str, Any],
    cache_dir: Path,
    trading_dates: pd.DatetimeIndex,
    instruments: dict[str, dict[str, Any]],
    share_transformations: dict[str, dict[str, Any]],
    transformation_source: dict[str, Any],
) -> tuple[pd.DataFrame, list[MembershipEvent], dict[str, Any]]:
    """Reverse official events from current anchors and fail closed on any gap."""

    start = pd.Timestamp(config["data"]["start_date"])
    requested_end = pd.Timestamp(config["data"]["end_date"])
    client = OfficialNoticeClient(cache_dir, int(config["data"]["network_timeout_seconds"]))
    configured_dates = {
        pd.Timestamp(value)
        for definition in config["data"]["universes"].values()
        for value in definition["verified_change_dates"]
    }
    required_notice_ids = {
        int(value)
        for definition in config["data"]["universes"].values()
        for value in definition.get("required_notice_ids", [])
    }
    csi_details = _discover_details(
        client, start, requested_end, configured_dates, required_notice_ids
    )
    sse_dates = {
        pd.Timestamp(value)
        for value in config["data"]["universes"]["sse50"]["verified_change_dates"]
    }
    required_sse_urls = {
        str(url)
        for definition in config["data"]["universes"].values()
        for url in definition.get("additional_official_sse_urls", [])
    }
    sse_details = _discover_sse_details(
        client, start, requested_end, sse_dates.union(configured_dates), required_sse_urls
    )
    if not csi_details or not sse_details:
        raise ValueError("Official index notice discovery returned no announcements.")

    all_rows: list[dict[str, Any]] = []
    all_events: list[MembershipEvent] = []
    applied_identity_events: list[dict[str, Any]] = []
    anchors: dict[str, Any] = {}
    symbol_codes: dict[str, list[str]] = {}
    for code, metadata in instruments.items():
        symbol = re.sub(r"\s+", "", str(metadata.get("symbol", "")))
        if symbol:
            symbol_codes.setdefault(symbol, []).append(code)
    for universe, definition in config["data"]["universes"].items():
        index_code = str(definition["official_index_code"])
        index_name = str(definition["official_index_name"])
        expected_count = int(definition["expected_constituents"])
        expected_dates = {pd.Timestamp(value) for value in definition["verified_change_dates"]}
        overrides = {
            str(key): pd.Timestamp(value)
            for key, value in definition.get("effective_date_overrides", {}).items()
        }
        anchor_date, members, anchor_meta = _official_anchor(client, index_code, expected_count)
        anchors[universe] = anchor_meta
        if anchor_date < requested_end:
            raise ValueError(f"Official {index_code} anchor predates requested end date.")

        parsed: dict[pd.Timestamp, MembershipEvent] = {}
        parse_failures: list[str] = []
        # Joint SSE/CSI adjustment notices often publish the same workbook on
        # both official sites.  CSI500 may therefore be recovered from either
        # official copy; duplicates must agree on the actual add/remove lists.
        if universe == "sse50":
            source_details = sse_details
        else:
            allowed_sse = set(map(str, definition.get("additional_official_sse_urls", [])))
            source_details = csi_details + [
                detail
                for detail in sse_details
                if str(detail.get("_source_url", "")) in allowed_sse
            ]
        for detail in source_details:
            try:
                effective = (
                    overrides.get(str(detail.get("id")))
                    or overrides.get(str(detail.get("_source_url", "")))
                    or overrides.get(f"publish:{detail.get('publishDate')}")
                )
                if effective is None:
                    effective = _effective_date(detail, trading_dates)
            except ValueError as exc:
                parse_failures.append(f"{detail.get('publishDate')} effective-date: {exc}")
                continue
            cleaned_source = _clean_html(str(detail.get("content", "")))
            if effective not in expected_dates:
                parse_failures.append(
                    f"{detail.get('publishDate')} unexpected-effective={effective.date()} title={detail.get('title')}"
                )
                continue
            try:
                add, remove, source_url, source_hash = _parse_event_codes(
                    client, detail, index_code, index_name, symbol_codes
                )
            except ValueError as exc:
                parse_failures.append(
                    f"{detail.get('publishDate')} codes for {effective.date()}: {exc}"
                )
                continue
            if not add and not remove:
                continue
            if len(add) != len(set(add)) or len(remove) != len(set(remove)):
                raise ValueError(f"Duplicate security in {index_code} event {effective.date()}.")
            event = MembershipEvent(
                index_code=index_code,
                announcement_date=str(detail["publishDate"]),
                effective_date=effective.date().isoformat(),
                add_list=sorted(add),
                remove_list=sorted(remove),
                source_url=source_url,
                source_sha256=source_hash,
            )
            if effective in parsed:
                existing = parsed[effective]
                if (
                    existing.add_list != event.add_list
                    or existing.remove_list != event.remove_list
                ):
                    raise ValueError(
                        f"Conflicting {index_code} events for {effective.date()}: "
                        f"{existing.source_url} ({len(existing.add_list)}/{len(existing.remove_list)}) "
                        f"vs {event.source_url} ({len(event.add_list)}/{len(event.remove_list)})."
                    )
            else:
                parsed[effective] = event

        details_by_id = {
            int(detail["id"]): detail
            for detail in csi_details
            if str(detail.get("id", "")).isdigit()
        }
        for specification in definition.get("derived_cascade_events", []):
            effective, event = _derive_official_cascade_event(
                client,
                specification,
                details_by_id,
                parsed,
                index_code,
            )
            if effective not in expected_dates:
                raise ValueError(
                    f"Derived {index_code} event has unconfigured date {effective.date()}."
                )
            if effective in parsed:
                existing = parsed[effective]
                if (
                    existing.add_list != event.add_list
                    or existing.remove_list != event.remove_list
                ):
                    raise ValueError(
                        f"Derived and directly parsed {index_code} events conflict at "
                        f"{effective.date()}."
                    )
            else:
                parsed[effective] = event

        missing_dates = sorted(value.date().isoformat() for value in expected_dates - set(parsed))
        if missing_dates:
            raise ValueError(
                f"Official {index_code} event coverage is incomplete; missing {missing_dates}; "
                f"parsed={sorted(value.date().isoformat() for value in parsed)}; "
                f"source_failures={parse_failures[:30]}."
            )

        identity_events: dict[pd.Timestamp, list[tuple[str, str, dict[str, Any]]]] = {}
        for predecessor, payload in share_transformations.items():
            successor = str(payload.get("successor", ""))
            if not predecessor.endswith((".XSHG", ".XSHE")) or not successor.endswith(
                (".XSHG", ".XSHE")
            ):
                continue
            effective = pd.Timestamp(payload["effective_date"])
            if start <= effective <= anchor_date:
                identity_events.setdefault(effective, []).append((predecessor, successor, payload))

        def overlaps_official(effective: pd.Timestamp, predecessor: str) -> bool:
            event = parsed.get(effective)
            return bool(
                event
                and predecessor in set(event.add_list).union(event.remove_list)
            )

        terminal_members = set(members)
        reverse_dates = sorted(set(parsed).union(identity_events), reverse=True)
        for effective in reverse_dates:
            event = parsed.get(effective)
            if event is not None:
                if not set(event.add_list).issubset(members):
                    absent = sorted(set(event.add_list) - members)
                    raise ValueError(
                        f"Reverse event has absent additions: {index_code} {effective.date()} {absent}."
                    )
                if set(event.remove_list).intersection(members):
                    present = sorted(set(event.remove_list).intersection(members))
                    raise ValueError(
                        f"Reverse event has already-present removals: {index_code} {effective.date()} {present}."
                    )
                members.difference_update(event.add_list)
                members.update(event.remove_list)
            for predecessor, successor, payload in reversed(identity_events.get(effective, [])):
                if overlaps_official(effective, predecessor):
                    continue
                if successor in members and predecessor not in members:
                    members.remove(successor)
                    members.add(predecessor)
            if len(members) != expected_count:
                raise ValueError(
                    f"{index_code} reverse count is {len(members)} at {effective.date()}, expected {expected_count}."
                )
        all_events.extend(parsed.values())

        # Reversing every event leaves the membership immediately before the
        # first in-sample event, which is the point-in-time set at sample start.
        current = set(members)
        sequence: list[tuple[pd.Timestamp, set[str], str]] = [(start, set(current), start.date().isoformat())]
        for effective in sorted(set(expected_dates).union(identity_events)):
            event = parsed.get(effective)
            announcement_date = effective.date().isoformat()
            changed = False
            if event is not None:
                current.difference_update(event.remove_list)
                current.update(event.add_list)
                announcement_date = event.announcement_date
                changed = True
            for predecessor, successor, payload in identity_events.get(effective, []):
                if overlaps_official(effective, predecessor) or predecessor not in current:
                    continue
                if successor in current:
                    raise ValueError(
                        f"{index_code} identity event collides with an existing member at "
                        f"{effective.date()}: {predecessor} -> {successor}."
                    )
                current.remove(predecessor)
                current.add(successor)
                changed = True
                applied_identity_events.append(
                    {
                        "index_code": index_code,
                        "effective_date": effective.date().isoformat(),
                        "predecessor": predecessor,
                        "successor": successor,
                        "event": payload.get("event"),
                        "share_conversion_ratio": payload.get("share_conversion_ratio"),
                        "source_file": transformation_source["source_file"],
                        "source_sha256": transformation_source["source_sha256"],
                        "parser_version": PARSER_VERSION,
                    }
                )
            if len(current) != expected_count:
                raise ValueError(f"{index_code} forward count failed at {effective.date()}.")
            if changed:
                sequence.append((effective, set(current), announcement_date))
        if current != terminal_members:
            raise ValueError(f"{index_code} reconstructed terminal members do not match official anchor.")
        for snapshot_date, snapshot_members, announcement_date in sequence:
            for code in sorted(snapshot_members):
                all_rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "provider_update_date": pd.Timestamp(announcement_date),
                        "universe": universe,
                        "code": provider_code(code),
                        "code_name": code,
                    }
                )

    events = sorted(all_events, key=lambda item: (item.effective_date, item.index_code))
    manifest = {
        "parser_version": PARSER_VERSION,
        "official_only": True,
        "anchors": anchors,
        "events": [asdict(item) for item in events],
        "event_count": len(events),
        "security_identity_events": applied_identity_events,
        "security_identity_event_count": len(applied_identity_events),
        "security_identity_source": transformation_source,
    }
    write_json(cache_dir / "membership-events.json", manifest)
    frame = pd.DataFrame(all_rows).sort_values(["snapshot_date", "universe", "code"])
    return frame.reset_index(drop=True), events, manifest

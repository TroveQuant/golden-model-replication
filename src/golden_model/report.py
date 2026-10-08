"""Self-contained HTML reporting and report-contract validation."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


COLORS = {
    "model1": "#2563eb",
    "model2": "#ea580c",
    "benchmark_equal_weight": "#64748b",
}


def _display(value: Any, metric: str | None = None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "unavailable"
    if metric and any(
        token in metric
        for token in (
            "return",
            "volatility",
            "drawdown",
            "transaction_cost",
            "tracking_error",
        )
    ):
        return f"{float(value):.2%}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape(str(value))


def _line_svg(
    equity: pd.DataFrame,
    value_column: str,
    *,
    normalize: bool,
    aria_label: str,
    width: int = 980,
    height: int = 330,
) -> str:
    series: dict[str, pd.Series] = {}
    for model, frame in equity.groupby("model", sort=True):
        values = frame.sort_values("date")[value_column].astype(float).reset_index(drop=True)
        if normalize:
            first = float(values.iloc[0])
            values = values / first if first != 0 else values
        series[str(model)] = values
    all_values = pd.concat(series.values(), ignore_index=True)
    if value_column == "drawdown":
        minimum = min(float(all_values.min()), 0.0)
        maximum = max(float(all_values.max()), 0.0)
    else:
        minimum = float(all_values.min())
        maximum = float(all_values.max())
    span = maximum - minimum if maximum > minimum else 1.0
    left, right, top, bottom = 54, 22, 24, 36
    polylines: list[str] = []
    legends: list[str] = []
    for legend_index, (model, values) in enumerate(series.items()):
        count = max(len(values) - 1, 1)
        points = []
        for index, value in enumerate(values):
            x = left + index / count * (width - left - right)
            y = top + (maximum - float(value)) / span * (height - top - bottom)
            points.append(f"{x:.2f},{y:.2f}")
        color = COLORS.get(model, "#111827")
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{" ".join(points)}" />'
        )
        legend_y = 17 + legend_index * 17
        legends.append(
            f'<line x1="70" y1="{legend_y}" x2="90" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3" />'
            f'<text x="96" y="{legend_y + 4}" font-size="11">{escape(model)}</text>'
        )
    zero_y = (
        top + (maximum / span) * (height - top - bottom)
        if value_column == "drawdown"
        else height - bottom
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(aria_label)}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff" />'
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#cbd5e1" />'
        + "".join(polylines)
        + "".join(legends)
        + f'<text x="{width-225}" y="18" font-size="11">Range: {minimum:.3f} to {maximum:.3f}</text>'
        + "</svg>"
    )


def _metric_table(metrics: dict[str, dict[str, Any]]) -> str:
    metric_order = [
        "start_date",
        "end_date",
        "observations",
        "final_nav",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "calmar",
        "max_drawdown",
        "max_drawdown_duration_sessions",
        "benchmark_total_return",
        "excess_total_return",
        "annualized_excess_return",
        "tracking_error",
        "information_ratio",
        "total_turnover",
        "total_transaction_cost",
        "trade_legs",
    ]
    headers = "".join(f"<th>{escape(model)}</th>" for model in metrics)
    rows = []
    for metric in metric_order:
        cells = "".join(
            f"<td>{_display(values.get(metric), metric)}</td>"
            for values in metrics.values()
        )
        rows.append(f"<tr><th>{escape(metric)}</th>{cells}</tr>")
    return f"<table><thead><tr><th>Metric</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _holdings_table(positions: pd.DataFrame) -> str:
    if positions.empty:
        return "<p>unavailable</p>"
    latest = positions.sort_values("date").groupby("model", sort=True).tail(1)
    rows = []
    for row in latest.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.model))}</td><td>{escape(str(pd.Timestamp(row.date).date()))}</td>"
            f"<td>{_display(row.sse50_weight)}</td><td>{_display(row.csi500_weight)}</td>"
            f"<td>{_display(row.cash_sleeve)}</td><td>{_display(row.gross_exposure)}</td>"
            f"<td>{_display(row.net_exposure)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Model</th><th>As of</th><th>SSE 50</th><th>CSI 500</th>"
        "<th>Cash sleeve</th><th>Gross exposure</th><th>Net exposure</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _trades_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<p>No executed target-weight changes in the sample.</p>"
    recent = trades.sort_values(
        ["execution_date", "model", "asset"], ascending=[False, True, True]
    ).head(24)
    rows = []
    for row in recent.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{escape(str(pd.Timestamp(row.signal_date).date()))}</td>"
            f"<td>{escape(str(pd.Timestamp(row.execution_date).date()))}</td>"
            f"<td>{escape(str(row.model))}</td><td>{escape(str(row.asset))}</td>"
            f"<td>{_display(row.from_weight)}</td><td>{_display(row.to_weight)}</td>"
            f"<td>{_display(row.transaction_cost, 'transaction_cost')}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Signal date</th><th>Execution date</th><th>Model</th>"
        "<th>Asset</th><th>From</th><th>To</th><th>Cost / NAV</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_report_html(
    config: dict[str, Any],
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> str:
    headline = str(config["strategy"]["headline_model"])
    validation = provenance.get("validation", {})
    migration = provenance.get("migration_validation", {})
    membership = provenance.get("historical_membership", {})
    preclose = provenance.get("preclose_validation", {})
    run = provenance.get("run", {})
    headline_metrics = metrics.get(headline, {})
    generated_payload = escape(
        json.dumps(
            {
                "metrics": metrics,
                "data_validation": validation,
                "migration_validation": {
                    "status": migration.get("status"),
                    "checks": migration.get("checks", {}),
                    "shared_securities": migration.get("shared_securities"),
                    "matched_rows": migration.get("matched_rows"),
                },
                "preclose_validation": preclose,
                "run": run,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    methodology = [
        ("Original material", "Model 1: sign(close - open); Model 2: sign(close - preclose)", "Implemented per security before aggregation", "adapted"),
        ("Original material", "Continuous red/green count capped at ±5 and ten-bucket weights", "Exact recovered ten-bucket values; zero/NaN resets the streak", "adapted"),
        ("Original material", "Three-day treatment of SSE 50 and CSI 500 sentiment", "Three-trading-session rolling mean, requiring all three observations", "matched"),
        ("Original material", "Relative signal: ΔE50 - ΔE500", "Positive: long 50/short 500; negative: reverse; zero: hold", "matched"),
        ("Engineering interpretation", "Target allocation", "+0.40/-0.40 index sleeves with a stated 0.20 cash sleeve", "adapted"),
        ("Engineering interpretation", "Signal and execution timing", "T close signal; next trading session close execution; new weights earn from the following interval", "adapted"),
        ("Public data substitute", "Wind/JQData inputs", "RQAlpha free monthly bundle for prices, calendar, suspensions and corporate actions", "adapted"),
        ("Public data substitute", "Historical constituents", "Event-sourced SSE/CSI official notices, applied on effective dates and checked at 50/500 members", "adapted"),
        ("Extension", "Evaluation benchmark", "50/50 buy-and-hold SSE 50 / CSI 500 close-index proxy", "extended"),
    ]
    methodology_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in methodology
    )
    missing_price_rows = validation.get("stock_price_missing_ohlc_rows", "unknown")
    halted_rows = validation.get("stock_price_nontrading_rows", "unknown")
    summary = (
        f"Headline {escape(headline)} ended at NAV {_display(headline_metrics.get('final_nav'))}, "
        f"with total return {_display(headline_metrics.get('total_return'), 'total_return')} and "
        f"maximum drawdown {_display(headline_metrics.get('max_drawdown'), 'max_drawdown')}."
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(config['project']['name'])}</title>
<style>
:root{{--ink:#172033;--muted:#5b677a;--line:#dce3eb;--blue:#2563eb;--orange:#ea580c}}
*{{box-sizing:border-box}}body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f3f6f9;color:var(--ink);line-height:1.55}}
main{{max-width:1160px;margin:0 auto;padding:28px}}.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:21px;margin:16px 0;overflow:auto}}
.notice{{border-left:5px solid var(--orange);background:#fff7ed}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.stat{{border:1px solid var(--line);border-radius:9px;padding:13px}}.stat span{{display:block;color:var(--muted);font-size:12px}}.stat strong{{font-size:20px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #e8edf2;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}small,.muted{{color:var(--muted)}}h1,h2{{margin-top:0}}h3{{margin-bottom:7px}}ul{{padding-left:20px}}svg{{width:100%;height:auto}}
</style>
</head>
<body><main>
<h1>{escape(config['project']['name'])}</h1>
<section class="card notice"><strong>复现口径：</strong>adapted public-data replication；不是原策略的严格复现。研究输出仅供方法验证，不构成投资建议。</section>
<section class="card"><h2>Executive Summary / 执行摘要</h2><p>{summary}</p>
<div class="grid"><div class="stat"><span>Status</span><strong>adapted</strong></div><div class="stat"><span>Sample</span><strong>{escape(str(config['data']['start_date']))} → {escape(str(provenance.get('actual_end', config['data']['end_date'])))}</strong></div><div class="stat"><span>Headline</span><strong>{escape(headline)}</strong></div><div class="stat"><span>Provider</span><strong>RQAlpha bundle</strong></div></div></section>
<section class="card"><h2>Headline Metrics / 核心指标</h2>{_metric_table(metrics)}<p class="muted">收益率均由同批次真实行情计算。Sharpe 的无风险利率为 {float(config['backtest']['risk_free_rate']):.2%}，年化交易日为 {int(config['backtest']['annualization_days'])}；benchmark/excess 为同日对齐的 50/50 指数基准。</p></section>
<section class="card"><h2>Figures and Result Tables / 图表</h2><h3>Normalized equity (each series starts at 1.0)</h3>{_line_svg(equity, 'nav', normalize=True, aria_label='Normalized equity curves')}<h3>Drawdown from running peak</h3>{_line_svg(equity, 'drawdown', normalize=False, aria_label='Drawdown curves')}</section>
<section class="card"><h2>Methodology Mapping / 方法映射</h2><table><thead><tr><th>Layer</th><th>Rule</th><th>Implementation</th><th>Status</th></tr></thead><tbody>{methodology_rows}</tbody></table></section>
<section class="card"><h2>Data and Assumptions / 数据与假设</h2>
<ul><li>Production provider: RQAlpha free monthly bundle {escape(str(provenance.get('provider_version','unknown')))}. BaoStock is excluded from production frames and used only for read-only migration comparison.</li>
<li>Extraction UTC: {escape(str(provenance.get('retrieved_at_utc','unknown')))}; requested coverage: {escape(str(config['data']['start_date']))} to {escape(str(config['data']['end_date']))}; observed coverage: {escape(str(validation.get('actual_start','unknown')))} to {escape(str(validation.get('actual_end','unknown')))}.</li>
<li>Rows: trade calendar {escape(str(validation.get('trade_date_rows','unknown')))}, constituents {escape(str(validation.get('constituent_rows','unknown')))}, stock prices {escape(str(validation.get('stock_price_rows','unknown')))}, index prices {escape(str(validation.get('index_price_rows','unknown')))}; historical stock identifiers {escape(str(validation.get('historical_constituent_codes','unknown')))}.</li>
<li>Stock rows with missing open/close/preclose: {escape(str(missing_price_rows))}; non-trading rows: {escape(str(halted_rows))}. Missing fields reset the per-security streak and non-trading rows are excluded from daily aggregation; no observations are interpolated.</li>
<li>Historical SSE 50 and CSI 500 membership is rebuilt from {escape(str(membership.get('event_count','unknown')))} official event records plus audited security-identity events. Changes apply on effective dates; each snapshot must contain exactly 50 or 500 members.</li>
<li>Preclose is reconstructed and audited from raw closes, RQAlpha cash/split records and cumulative factors. Classified complex actions or suspension resets use the bundle's explicit comparable reference; controlled fallbacks in sample: {escape(str(preclose.get('controlled_fallback_count','unknown')))}; unexplained differences: {escape(str(preclose.get('unexplained_difference_count','unknown')))}.</li>
<li>T close generates the signal; T+1 close executes. New weights earn returns only after that close. One-way cost is {float(config['backtest']['one_way_cost_bps']):.2f} bps × sum of absolute target-weight changes.</li></ul></section>
<section class="card"><h2>Migration Validation / 数据迁移验收</h2><div class="grid"><div class="stat"><span>Status</span><strong>{escape(str(migration.get('status','unknown')))}</strong></div><div class="stat"><span>Shared securities</span><strong>{escape(str(migration.get('shared_securities','unknown')))}</strong></div><div class="stat"><span>Matched rows</span><strong>{escape(str(migration.get('matched_rows','unknown')))}</strong></div><div class="stat"><span>50/500 signal agreement</span><strong>{_display(migration.get('relative_50_500_signal_agreement'), 'return')}</strong></div></div><p class="muted">BaoStock cache is read-only comparison evidence. It is never merged into production market data.</p></section>
<section class="card"><h2>Current Holdings / 当前目标持仓</h2>{_holdings_table(positions)}<p class="muted">These are the final backtest target sleeves, not a live allocation recommendation.</p></section>
<section class="card"><h2>Recent Transactions / 近期交易</h2>{_trades_table(trades)}</section>
<section class="card"><h2>Fidelity Gaps and Limitations / 保真差异与限制</h2><ul>
<li>原始材料未提供可独立核验的完整生产代码、逐日历史输出或精确数据库版本，因此本项目只能标记为公开数据代理的策略适配复现。</li>
<li>Wind/聚宽口径由 RQAlpha 免费月度 bundle 与上交所/中证指数官方调样公告替代；这会形成供应商价格、复权与参考价口径差异。</li>
<li>指数收盘价是可交易代理；未模拟期货保证金、融券可得性、印花税、冲击成本、成交容量或现金利息，基准也不含交易成本。</li>
<li>历史成分通过官方公告事件式重建，包括临时调整和生效日；公告缺失、解析冲突或成员数异常都会失败关闭，不会回填当前成分。</li>
<li>免费 bundle 对复杂配股、重组和长期停牌参考价的结构化条款并不总是完整，因此明确记录受控 reference fallback；未知差异不会静默放行。</li>
</ul></section>
<section class="card"><h2>Reproducibility / 可复现性</h2><ul>
<li>Commit SHA: <code>{escape(str(run.get('commit_sha','unknown')))}</code></li><li>Generated UTC: {escape(str(run.get('generated_at_utc','unknown')))}</li>
<li>Runtime: {escape(str(run.get('runtime','unknown')))}</li><li>Command: <code>{escape(str(run.get('command','python run.py all')))}</code></li>
<li>Configuration: <code>config/default.yaml</code>; resolved copy: <code>results/{escape(str(config['outputs']['resolved_config']))}</code>.</li>
</ul><p>All metrics, figures, holdings and transactions in this file are generated from the same run. No test fixture, random series, synthetic fallback, external script, stylesheet or image is embedded.</p>
<small>Embedded audit payload: <span id="audit-payload">{generated_payload}</span></small></section>
</main></body></html>"""


def validate_report(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["outputs"]
    required = [
        outputs["equity_curve"],
        outputs["positions"],
        outputs["trades"],
        outputs["sentiment"],
        outputs["signals"],
        outputs["metrics"],
        outputs["migration_validation"],
        outputs["provenance"],
        outputs["resolved_config"],
        outputs["report_payload"],
        outputs["report_html"],
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing report artifacts: {missing}")
    html = (output_dir / outputs["report_html"]).read_text(encoding="utf-8")
    for forbidden in ("<script src=", "<link rel=", "http://", "https://"):
        if forbidden.lower() in html.lower():
            raise ValueError(f"Report is not self-contained; found {forbidden!r}.")
    required_text = [
        config["project"]["name"],
        "adapted public-data replication",
        "不构成投资建议",
        "T+1 close executes",
        "Headline Metrics",
        "Drawdown from running peak",
        "Methodology Mapping",
        "Data and Assumptions",
        "Migration Validation",
        "Current Holdings",
        "Recent Transactions",
        "Fidelity Gaps and Limitations",
        "Reproducibility",
        'id="audit-payload"',
    ]
    absent = [text for text in required_text if text not in html]
    if absent:
        raise ValueError(f"Report missing required disclosures: {absent}")
    if html.count("<svg") < 2:
        raise ValueError("Report must embed both equity and drawdown figures.")
    metrics = json.loads((output_dir / outputs["metrics"]).read_text(encoding="utf-8"))
    equity = pd.read_csv(output_dir / outputs["equity_curve"])
    if set(metrics) != set(equity["model"].unique()):
        raise ValueError("Metrics and equity curve model sets differ.")
    return {"artifacts": len(required), "models": len(metrics), "self_contained": True}

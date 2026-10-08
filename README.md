# 金巅模型策略适配复现版本（公开数据代理版）

本仓库是 **adapted public-data replication**，不是原始生产策略的严格复现。它将现有“金巅模型”材料整理为可运行、可审计、可自动复现的完整链路；正式行情来自 RQAlpha 免费月度 bundle，历史成分由上交所与中证指数官方公告按生效日事件式重建。项目不使用随机、模拟、插值或写死收益数据。

> 仅用于研究复现与工程验证，不构成投资建议。

## 策略与时序

```text
RQAlpha日线/交易日历/停牌与公司行为
  + 上交所/中证指数官方调样公告
  → PIT历史成分 → 模型1/模型2情绪 → 3日算术平均
  → 50/500相对强弱信号 → 目标权重
  → T+1收盘执行 → 成本 → 净值/持仓/交易/指标/HTML报告
```

- 模型1输入为 `close - open`；模型2输入为 `close - preclose`。
- 连续红绿状态封顶为5日，并映射到十档固定权重。
- 上证50与中证500分别做3个交易日算术平均。
- 信号为 `[ΔE50(T) - ΔE500(T)]`；正向时上证50 `+40%`、中证500 `-40%`，负向时反转，现金袖套 `20%`。
- T日收盘后生成信号，T+1交易日收盘执行；新仓位只从下一收盘区间贡献收益。
- 单边交易成本为 `5 bps × Σ|目标权重 - 执行前权重|`。

详细公式见 [docs/methodology.md](docs/methodology.md)。

## 数据口径

- 行情主源：RQAlpha `6.4.0` 免费月度 bundle；股票日线、指数日线、交易日历、`suspended_days` 和公司行为均来自同一 bundle。
- 历史成分：上证50与中证500官方定期/临时调样公告，按 `effective_date` 应用；每条事件保留来源 URL、SHA-256 和解析器版本。
- 固定请求区间：`2015-04-16` 至 `2026-09-30`，另取14个自然日预热；当前 bundle 实际截止日为 `2026-09-30`。
- 样本覆盖：1458只历史成分证券、2797个交易日、3,601,967条股票日线。
- `preclose`：普通日使用上一真实收盘；明确现金分红/拆并股用公司行为公式；仅对已分类的复杂公司行为或停牌重置使用 bundle 明示参考值，无法解释的差异直接失败。
- 停牌：只使用 `suspended_days.h5`，并区分上市前、停牌、正常交易、退市后和数据缺失。
- BaoStock：现有1092只缓存仅用于只读迁移校验，不进入正式行情、成分或回测。

官方公告解析、成员数量、价格字段、停牌和公司行为检查均为 fail-closed；不会用当前成分回填历史。原始 bundle、公告缓存和 BaoStock 缓存不提交 Git，使用者仍需遵守数据提供方条款。

## 安装与运行

推荐 Python 3.11/3.12：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python run.py validate
python run.py fetch
python -m pytest -q
python run.py all
python run.py validate-report
```

`python run.py fetch --refresh` 会从公开源重新下载 RQAlpha bundle 与官方公告；`backtest`、`report`、`all` 都通过统一生产管线重建结果。Windows PowerShell 可用 `./.venv/Scripts/python.exe` 替换 `python`。

## 当前真实回测结果

以下数值由 `python run.py all` 使用上述真实数据生成，区间为 `2015-04-16` 至 `2026-09-30`，2788个回测观测：

| 组合 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 最终净值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 模型1 | -14.0981% | -1.3647% | 7.5947% | -0.1429 | -36.8330% | 0.859019 |
| 模型2 | -35.1606% | -3.8418% | 7.6071% | -0.4768 | -49.3854% | 0.648394 |
| 50/50指数基准 | -4.0283% | -0.3711% | 21.2311% | 0.0893 | -49.6831% | 0.959717 |

模型1/模型2累计换手分别为1328.8/1327.2，按5 bps口径累计成本分别为0.6644/0.6636。负收益是本次公开数据适配复现的真实结果，没有优化或隐藏。

迁移校验覆盖1092只共同证券、1,394,540条匹配日线；`sign(close-open)` 一致率100%，`sign(close-preclose)` 一致率99.99749%，停牌一致率99.99993%，50/500最终信号一致率99.37231%。完整统计在 [audit/migration_validation.json](audit/migration_validation.json)。

## 输出与自动化

`results/` 中的正式产物全部由统一管线重建：

- `equity_curve.csv`、`positions.csv`、`trades.csv`
- `metrics.json`、`report.html`
- `sentiment.csv`、`signals.csv`
- `data_provenance.json`、`migration_validation.json`
- `resolved_config.yaml`、`report.json`

`.github/workflows/backtest.yml` 在干净 Ubuntu runner 上安装固定依赖、下载真实 bundle、重建官方 PIT 成分、运行测试和完整回测、校验报告、上传 artifact，并将本次结果回推 `main`。CI 不使用本机缓存或 BaoStock 行情；它只验证已提交且通过阈值的迁移审计门禁，然后从 RQAlpha/官方公告重新生成正式结果。

## 项目边界

- 原材料缺少完整生产代码、数据库快照和逐日目标输出，因此只能称为策略适配复现。
- RQAlpha 与原 Wind/聚宽口径无法保证逐点一致；所有可确认差异均披露并受迁移阈值约束。
- 指数收盘价是可交易代理，未模拟期货保证金、融券可得性、税费、冲击成本、容量或现金利息。
- 50/50基准、风险指标与报告属于工程评估扩展，不是原策略规则。

详见 [保真差异](docs/fidelity-gaps.md) 与 [可复现说明](docs/reproducibility.md)。

## License

代码以 MIT License 发布。RQAlpha、公开公告和市场数据保留各自的许可与使用条款，本仓库不再分发原始数据文件。

# 可复现与自动化说明

## 本地复现

依赖固定在 `requirements.txt`。标准顺序为：

```bash
python -m pip install -r requirements.txt
python run.py validate
python run.py fetch --refresh
python -m pytest -q
python run.py all
python run.py validate-report
```

`fetch --refresh` 下载并校验 RQAlpha bundle、获取官方公告、重建历史成分并写入 SHA-256 provenance。无 `--refresh` 时只复用模式版本与哈希检查通过的缓存。正式结果不读取仓库旧 `results/` 作为输入。

## 迁移门禁

本地存在 `data/cache/_partial` 时，管线必须用1092只已校验 BaoStock 缓存重新执行逐点迁移比较，随后原子更新 `audit/migration_validation.json`。BaoStock 数据始终只读且不进入生产 frame。

公开 CI 不能也不得携带该本地缓存，因此使用已提交的认证审计作为门禁。只有当审计状态、四项检查、最小参考证券数、生产 provider 和截止日全部匹配时才放行；审计缺失或字段异常会直接失败。放行后，CI 仍从空缓存重新下载 RQAlpha 与官方公告并完整重建回测。

## GitHub Actions

`.github/workflows/backtest.yml` 的关键步骤均无 `continue-on-error`：

1. 检出 `main` 并清空历史 `results/`；
2. 安装固定依赖；
3. 配置校验并从公开源执行 `fetch --refresh`；
4. 执行全部测试；
5. 执行 `python run.py all` 与报告验证；
6. 上传完整 `results/` 和迁移审计 artifact；
7. 将本次生成结果提交并回推 `origin/main`。

工作流使用 `contents: write`，回推提交带 `[skip ci]` 防止递归触发。任何真实数据、测试、报告、artifact 或推送步骤失败，workflow 都不会得到 success。

## 原始数据与公开边界

以下内容被 `.gitignore` 排除：RQAlpha bundle、官方公告下载缓存、BaoStock缓存、虚拟环境、凭据、同步工具目录、原始研究文档/表格/图片。仓库仅发布代码、配置、审计摘要和派生结果。数据服务及官方文件的使用者需自行遵守其许可与条款。

HTML 报告不加载外部脚本、字体、图片或样式；`python run.py validate-report` 检查必需产物、模型集合、核心披露、图表内嵌与外部依赖。

<div align="center">
  <h1>chinese-evidence-retrieval-lab</h1>
  [**English**](./README.md) | **中文**
</div>
<br>

> 一个可复现的 PyTorch 实验室，用于评估证据感知型研究助理所使用的中文两阶段检索策略。

`chinese-evidence-retrieval-lab` 是 [CountyResearchAI](https://github.com/OrdoAbChao7/CountyResearchAI) 的实验配套项目。在将任一策略集成到上游研究助理之前，它先在公开的中文排序基准上评估该策略是否提升了段落的**相关性**。

**本项目不声称验证县域研究报告的事实正确性，亦不会将 CountyResearchAI 的私有数据发送至任何第三方服务。** 实验围绕公开基准数据、固定的训练/开发划分、版本化清单以及可重复的度量设计。

## 研究问题

给定中文查询与段落语料库，一个两阶段系统——先进行稠密检索，再用交叉编码器重排——是否能相较于词法与预训练稠密检索基线获得更好的排序质量？

主要研究使用 [T2Ranking](https://github.com/THUIR/T2Ranking)，这是一套在 Apache-2.0 许可下发布的中文段落排序公开基准，提供官方的训练/开发查询划分和检索 qrels。该基准同时覆盖一阶段检索与二阶段重排，贴合证据助理“先检索再重排”的工作流。[^^t2ranking]

## 实验阶梯

| ID | 系统 | 训练 | 目的 | 状态 |
|---|---|---:|---|---|
| E0 | BM25 | 否 | 词法基线 | 已搭建 |
| E1 | 预训练中文双编码器 | 否 | 语义基线 | 已搭建 |
| E2 | E1 + 批内对比学习 | 是 | 建立透明的 PyTorch 训练循环 | 已搭建 |
| E3 | E2 + 困难负例训练 | 是 | 检验困难负例是否提升检索 | 计划中 |
| E4 | 在 E3 Top-100 上 + 交叉编码器重排 | 可选 | 测试完整的两阶段排序栈 | 计划中 |

在生成带版本号的运行结果之前，不宣称任何指标。首次提交时，`artifacts/metrics/` 目录有意保持为空。

## 评测协议

严格保留 T2Ranking 的官方训练/开发划分。训练查询、训练 qrels 与负例文件绝不用于开发集评估。测试套件会检查查询重叠、正负样本冲突、指标行为，以及仅 CPU 的冒烟训练路径。

| 指标 | 报告原因 |
|---|---|
| nDCG@10 | 主要排序指标；奖励正确的排序并支持分级相关性。 |
| MRR@10 | 衡量首个相关结果出现的速度。 |
| Recall@50 / Recall@100 | 检验一阶段检索是否将相关证据传递给后续重排。 |
| Latency | 使质量与成本的权衡显性化。 |
| 跨种子均值 ± 标准差 | 防止挑选单次有利结果。 |

## 快速开始

唯一必须的首次运行是小型 `smoke` 预设。它会从**本地下载**的 T2Ranking 副本中创建一个紧凑且可确定的子集，并不会向任何地方提交数据。

```bash
python -m venv .venv
# macOS/Linux
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -e ".[dev,train]"

# Place the official T2Ranking files under data/raw/t2ranking/ first.
python scripts/prepare_subset.py \
  --source-dir data/raw/t2ranking/data \
  --output-dir data/processed/smoke \
  --preset smoke \
  --seed 2026

pytest
```

数据下载、许可与文件布局详见 [data/README.md](data/README.md)。完整协议见 [docs/experiment_protocol.md](docs/experiment_protocol.md)。

## 仓库结构

```text
src/evidence_retrieval/  # dataset parsing, models, losses, index, metrics, analysis
configs/                 # E0–E4 declarative experiment settings
scripts/                 # subset preparation, training/evaluation entry points
artifacts/               # compact manifests, metrics, figures and error analyses
tests/                   # leakage, metric, collator, loss and smoke-training tests
docs/                    # protocol, data card and model card
data/README.md           # data access and licensing; raw data is never committed
```

## 可复现性承诺

每一条实验结果都必须包含：精确的配置文件、随机种子、Git 提交号、Python/依赖包版本、设备信息、数据清单哈希、计时信息与输出运行文件。缺少这些元数据的结果被视为探索性，不会作为最终实验结果进行报告。

## 与 CountyResearchAI 的关系

只有在开发集上表现稳定且已记录失败案例的检索器，才会被考虑提供可选的 CountyResearchAI 适配器。任何未来的上游集成都必须保留模型版本、索引版本、top-k 证据列表与来源 URL。基准上的相关性并不保证生成的报告正确；报告结论仍需对来源与时间范围进行审查。

## 许可证

本代码以 [MIT License](LICENSE) 发布。T2Ranking 以 Apache-2.0 单独授权；使用时请查阅其上游文档并正确引用该基准。[^^t2ranking]

[^t2ranking]: Xie 等人，[T2Ranking: A Large-scale Chinese Benchmark for Passage Ranking](https://github.com/THUIR/T2Ranking/)。官方仓库记录了数据文件、官方 train/dev 划分、检索/重排任务以及 Apache-2.0 许可。

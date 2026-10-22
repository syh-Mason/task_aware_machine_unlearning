# 三条机器遗忘路线：报告与阅读顺序

**本轮交付：三份独立研究报告，每份均有可编辑 Word、PDF 和 Markdown，附三张流程图、14 项教学数值校验、来源清单。**

[一次下载完整 ZIP](https://github.com/syh-Mason/task_aware_machine_unlearning/blob/reports/unlearning-three-methods/exports/unlearning_three_reports.zip?raw=true)

| 报告 | 代表论文 | Word | PDF |
|---|---|---|---|
| SISA：分组重训练 | Bourtoule et al., S&P 2021 | [下载](报告一_Exact遗忘_SISA分组重训练.docx?raw=true) | [阅读](报告一_Exact遗忘_SISA分组重训练.pdf) |
| 影响函数与认证删除 | Guo et al., ICML 2020 | [下载](报告二_Approximate遗忘_InfluenceFunction与认证删除.docx?raw=true) | [阅读](报告二_Approximate遗忘_InfluenceFunction与认证删除.pdf) |
| 持续学习启发的 SSD | Foster et al., AAAI 2024 | [下载](报告三_Approximate遗忘_持续学习启发的SSD与模型分层.docx?raw=true) | [阅读](报告三_Approximate遗忘_持续学习启发的SSD与模型分层.pdf) |

## 先读综述，再读代表论文

1. [Nguyen 等：A Survey of Machine Unlearning](https://arxiv.org/abs/2209.02299)：2022 预印本；作者确认的 [2025 ACM TIST 版](https://doi.org/10.1145/3749987)。
2. [Jiang 等：Machine Unlearning: A Survey，ACM CSUR](https://doi.org/10.1145/3603620)：作为第二份总览。不要与 Xu 等同名预印本混用。
3. 按 SISA → 影响函数 → SSD 顺序读三份报告，最后比较电网迁移方案。

目前实时学术引文接口受限，不能给出独立核实的当前排名。[历史引用线索](citation_snapshot_note.md) 只保留第三方 2026-05-10 快照的原始展示值与局限，不充当当前引用量。

## 三条路线的关键区别

| 路线 | 删除机制 | 参考与保证边界 | 电网项目中的第一步 |
|---|---|---|---|
| SISA | 受影响分片从干净检查点重训 | 同一分片训练流程；完整状态和数据依赖需要审计 | 独立整 CNN 分片或条件于固定核心的多头分片 |
| Influence / Newton | 用梯度变化和曲率修正参数 | 一般局部近似；固定特征二次目标可精确；认证另需假设 | 与现有固定核心仿射头直接重拟合核对 |
| SSD | 按删除集重要性选择并衰减参数 | 持续学习启发的经验近似，没有自动认证 | head-only 参数白名单与梯度平方重要性 |

## 阅读证据与执行状态

- Guo：已取得并读取 PMLR 正式全文 11 页及作者实现。
- SISA、SSD：作者源码与 README 已读，全文访问 403；报告明确为代码核对与独立推导，未冒充全文精读。
- 两篇综述：已核对公开文献索引；全文本轮未取得。
- 14 项合成教学检查通过，0 失败；不是论文基准、电网实验或隐私认证。
- 本轮不重新训练模型、不运行新的电网 Cost、不修改旧实验记录。电网迁移表均为待实施计划。

## 复算与重新生成

仓库根目录执行：

```bash
python experiments/verify_unlearning_reports.py
python reports/build_unlearning_three_reports.py
```

数值检查仅需 NumPy。文档生成还需 python-docx、PyMuPDF、Pillow，以及 pandoc、Graphviz、soffice 和 Noto CJK 字体。
首次环境中的来源缓存保存在工作区工具目录；其他环境可使用已提交的来源清单核对本地证据，不要求重新下载论文。

独立分支：`reports/unlearning-three-methods`；不改 master，不自动合并。

[来源与访问状态](source_manifest.json) · [数学检查](numerical_checks.json) · [文档检查](document_validation.json) · [SHA256](SHA256SUMS)

日期：2026-10-22（北京时间）。

# 两次实验报告与组会汇报

共三份主报告：两轮实验各一份正式实验报告，以及一份综合组会汇报。整理日期：2026-10-17。
组会汇报另附逐页讲稿与答疑备忘。文档基于已提交实测记录编制，没有重新训练模型或重跑实验。

| 材料 | 可编辑版本 | 阅读版本 |
| --- | --- | --- |
| 实验报告一：单样本温度扰动，9页 | [Word](01_single_sample/实验报告一_单样本温度扰动.docx) | [PDF](01_single_sample/实验报告一_单样本温度扰动.pdf) · [Markdown](01_single_sample/实验报告一_单样本温度扰动.md) |
| 实验报告二：节点成本诊断与多日期验证，10页 | [Word](02_cost_attribution/实验报告二_节点成本诊断与多日期验证.docx) | [PDF](02_cost_attribution/实验报告二_节点成本诊断与多日期验证.pdf) · [Markdown](02_cost_attribution/实验报告二_节点成本诊断与多日期验证.md) |
| 组会汇报：两轮实验研究进展，15页 | [PPT](03_group_meeting/组会汇报_CNN输入影响路径.pptx) | [PDF](03_group_meeting/组会汇报_CNN输入影响路径.pdf) |
| 组会逐页讲稿与答疑备忘，6页 | [Word](03_group_meeting/组会汇报_讲稿.docx) | [PDF](03_group_meeting/组会汇报_讲稿.pdf) · [Markdown](03_group_meeting/组会汇报_讲稿.md) |

**[下载全部文件 ZIP](https://github.com/syh-Mason/task_aware_machine_unlearning/blob/reports/cnn-experiments-group-meeting/exports/cnn_experiment_reports.zip?raw=true)**

组会PPT建议用于12—15分钟汇报。每页已写入演讲者备注；没有PPT软件也可直接阅读PDF和讲稿。

## 两份实验报告的范围

- 实验报告一：固定参考样本的五档训练标准差温度扰动、输入/输出结构、两阶段成本路径、四模型响应、完整20组指标、恢复与求解验证、局限和复算命令。
- 实验报告二：输出节点隔离/复位诊断、参考点抽样Shapley、27个新分层日期加参考点的28日期对照、配对经济性、明确反例、全部统计分母与后续研究边界。
- 组会汇报：串联研究问题、方法、两轮证据与下一步；区分绝对成本和响应幅度，保留反例，不把本批节点峰值一致扩大为普遍规律。

实验一记录为49项测试通过；实验二是扩展后的累计57项，包含前者，不能相加。报告内已明确这一点。
“重拟合头”沿用原项目冻结卷积核心、仅拟合最后910个仿射参数的协议。

## 数据与溯源

- [evidence/](evidence/)：支撑报告数值表格和图表的原始CSV/JSON副本。
- [source_manifest.json](source_manifest.json)：实验提交、源文件SHA256、工具版本和图表来源说明。
- [document_validation.json](document_validation.json)：文档页数、表格与原CSV一致性、PPT备注与导出完整性检查。
- [SHA256SUMS](SHA256SUMS)：本文件夹全部交付文件的校验清单。
- [assets/](assets/)：为报告和投影可读性重新排版的图表，均由保存的实际数据绘制。

实验来源为 `b5fd9f1`（单样本）和 `b527981`（节点诊断及多日期）。完整研究快照、全部求解轨迹及复算脚本仍在仓库 `research/` 和 `experiments/` 中；报告包不重复携带全部年度数据或运行环境。

本轮文档源码在仓库的 `reports/build_reports.py`、`reports/render_reports.py` 和 `reports/validate_and_export_reports.py`。在完整仓库根目录，使用具备 python-docx、python-pptx、matplotlib、pandas、Pillow、PyMuPDF 及 LibreOffice 的文档工具环境运行：

```bash
python reports/build_reports.py
python reports/render_reports.py
python reports/validate_and_export_reports.py
```

上述命令仅生成、渲染和核验文档，不运行模型训练或Cost实验。文档工具与原实验 `.venv` 分开使用，未改变项目实验依赖。

# Task-Aware Machine Unlearning：分层导图

**一张总导图 + 八张分层详图。按研究逻辑重新组织，非原文章节号。**

原论文 HTML/PDF 目前仍被网络代理以 403 拦截。本版基于作者 README 摘要、已核对代码、标准数学推导及明确单列的扩展实验。原论文完整目录、公式号、定理条件与证明尚未逐项核验，不能将这八层称为论文原目录。

- [下载全部材料 ZIP](../../exports/tamu_layered_mindmaps.zip?raw=true)
- [查看九页 A3 矢量导图合集 PDF](TAMU_分层导图合集.pdf)
- [下载分层阅读说明 Word](TAMU_分层阅读说明.docx?raw=true) · [十一页说明 PDF](TAMU_分层阅读说明.pdf)
- [在线阅读逐层解释](分层阅读说明.md)
- [总导图高清 PNG](figures/00_overview.png) · [可放大 SVG](figures/00_overview.svg)
- [离线交互 HTML](TAMU_交互导图.html?raw=true)：下载后用本地浏览器打开，支持切层、缩放、拖动与搜索。无需联网和外部依赖。
- [可编辑 FreeMind 格式](TAMU_可编辑导图.mm?raw=true) · [Mermaid 总图源文件](总导图.mmd) · [结构化 JSON](mindmap.json)

![八层总导图](figures/00_overview.png)

| 学习层 | 核心问题 | 详图 |
|---|---|---|
| 01 研究动机与问题 | 为什么需要删数据影响，还要保留模型效用？ | [PNG](figures/01_layer.png) / [SVG](figures/01_layer.svg) |
| 02 数据与预测模型 | 数据、核心、预测头、参数和尺度分别是什么？ | [PNG](figures/02_layer.png) / [SVG](figures/02_layer.svg) |
| 03 直接遗忘与影响函数 | 为什么能用梯度、Hessian、Newton 更新？ | [PNG](figures/03_layer.png) / [SVG](figures/03_layer.svg) |
| 04 PAMU | 怎样按 MSE/MAPE 为剩余样本调权？ | [PNG](figures/04_layer.png) / [SVG](figures/04_layer.svg) |
| 05 两阶段运行成本 | 预测怎样影响调度及 cost1、cost2？ | [PNG](figures/05_layer.png) / [SVG](figures/05_layer.svg) |
| 06 TAMU | Cost 梯度怎样返回模型，再形成样本分数和权重？ | [PNG](figures/06_layer.png) / [SVG](figures/06_layer.svg) |
| 07 实验与证据分层 | 作者研究范围、复现与你的温度扩展如何区分？ | [PNG](figures/07_layer.png) / [SVG](figures/07_layer.svg) |
| 08 结论与边界 | 能说明什么，哪些保证与比较还没有证据？ | [PNG](figures/08_layer.png) / [SVG](figures/08_layer.svg) |

每层有三个主分支、九个关键点，并在说明中给出通俗解释、关键关系、源码入口和先修链接。你的两次温度实验明确标为扩展研究，不冒充原论文实验。

## 文件使用与检查

PNG 方便插入汇报；SVG 和图集 PDF 适合放大；Word 方便改写讲稿；`.mm` 可导入 FreeMind 或兼容软件；每张图还附 Graphviz `.dot` 源文件。

已检查九张图的节点数、PDF页数和文字边界、DOCX/ZIP结构、编辑文件XML以及引用的扩展实测数值。全部图保留来源标记。交互脚本已通过 Node 语法检查；当前没有可用 Chromium，未进行真实浏览器交互验收，详见 `validation.json`。

本轮只组织学习内容和绘图，没有重新训练、重新计算实验或改动数学模型。`source_manifest.json` 记录来源与访问状态，`SHA256SUMS` 提供文件校验。

完整仓库中重建：

```bash
python reports/build_tamu_layered_mindmaps.py
```

构建需要 Graphviz、python-docx、Pillow、PyMuPDF、LibreOffice 和 Noto CJK 字体。科学计算环境 `.venv` 未修改。

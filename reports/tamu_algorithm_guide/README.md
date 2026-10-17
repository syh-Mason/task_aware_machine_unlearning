# TAMU 算法学习讲义：代码核对版

**来源限制：当前环境未取得目标论文及铺垫论文全文。** arXiv、IEEE、PMLR 等站点访问被网络代理以 403 拒绝。本版基于已经读取的作者原仓库、配套库说明、作者可微凸优化笔记和标准数学推导；不是逐页核对原文的精读定稿。Word 首页与附录已写明这一点，没有虚构原论文公式号、算法号或定理证明。

- [下载可编辑 Word](TAMU算法精读_先修论文与逐步推导.docx?raw=true)
- [阅读 PDF（21 页）](TAMU算法精读_先修论文与逐步推导.pdf)
- [下载 Word、PDF、Markdown 和数学验证脚本 ZIP](../../exports/tamu_algorithm_guide.zip?raw=true)
- [在线阅读文字版](算法精读与推导讲义.md)

同一份 Word 内包含：

1. 五篇核心/边界论文、两份数学资料、五篇扩展材料的链接与阅读顺序，说明每篇用于理解哪个环节。
2. 影响函数、删除符号与缩放、剩余集 Newton 更新，以及二次预测头精确性条件的推导。
3. PAMU/TAMU 的样本评分、实际展开锚点、L1/Linf 权重 LP 与加权预测头拟合。
4. 当前仓库完整两阶段目标与约束、KKT 求导、伴随法和 Cost 链式法则。
5. 固定到提交的代码位置、温度扰动实验的联系、方法边界，以及取得原文后的补核验清单。

公式为可编辑 Office Math，含 234 个数学对象（包括行内数学），并非公式截图。Word 有 46 个不同的外部超链接目标，另有可点击的导读目录。

## 验证及使用范围

`verify_derivations.py` 的 10 项检查通过，覆盖教学例子的删除、影响函数有限差分、加权拟合、权重预算、KKT/伴随导数与非对称成本。它们不是新的 TX-123BT 实验，也不是对尚未取得的论文定理的验证。

文中引用的四模型参考结果和 108 个非零扰动场景统计，已与原有 CSV 核对；没有重新训练模型。PDF 各页已检查，表格与数学模型已人工抽查，未发现超出页面的文字。

`source_manifest.json` 记录已读来源、文件散列及全文未取得状态；`document_validation.json` 记录文档检查；`SHA256SUMS` 可校验文件。

只验证教学推导可在解压文件夹中执行：

```bash
python verify_derivations.py
```

需要 NumPy。完整重建需要检出本仓库、保留两个既有研究目录，并使用文档工具环境执行：

```bash
python reports/build_tamu_algorithm_guide.py
```

构建器依赖 python-docx、NumPy、Matplotlib、PyMuPDF、Pillow、Pandoc、LibreOffice 与 Noto CJK 字体，不修改科学计算环境 `.venv`。

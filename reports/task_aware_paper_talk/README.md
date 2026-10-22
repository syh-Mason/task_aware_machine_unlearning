# Task-Aware 论文算法汇报｜10–13 分钟工作稿

**当前是可审阅工作稿，尚不是原文核对终稿。** 用户指定的 v2 论文与 NSE 指南页面被网络代理403拦截。

[下载完整包](https://github.com/syh-Mason/task_aware_machine_unlearning/blob/slides/task-aware-paper-mit/exports/task_aware_paper_talk.zip?raw=true) ·
[下载PPT](TaskAware_论文算法解析_10-13分钟_待原文核验.pptx?raw=true) · [阅读PDF](TaskAware_论文算法解析_10-13分钟_待原文核验.pdf)

## 已完成

- 12张主讲页 + 5张备份页，计划11分50秒（不是实测语速）。
- 重点：直接遗忘、PAMU、权重LP、TAMU、两阶段Cost、KKT隐式求导、iHVP样本评分、最终带权MSE更新。
- 每页备注内附讲稿、时长、公式源码与来源。
- 作者仓库原始流程图原样插入并标注来源，未冒称已确认的v2图号。
- [待补图方法文本](待补图方法论文本.md)，含3份可交给制图工具的说明；没有调用图像生成。
- [讲稿与控时](逐页讲稿与控时.md)、[MIT设计核对](MIT指南设计核对.md)、[原文核对清单](原文核对清单.md)。

## 仍然缺少的材料

**请上传 arXiv:2308.14412v2 PDF。** 正式稿需要核对原文公式、梯度定理和结果图。当前没有原文实验图，只有作者仓库原图。

How to Speak 已取得 MIT 官方课程页源码，未观看全视频；NSE Slideshow 正文未取得，视觉设计为归纳适配，不声称逐条验证其原页标准。

## 文件

- `TaskAware_论文算法解析_10-13分钟_待原文核验.pptx` / `.pdf`：演示工作稿。
- `逐页讲稿与控时.docx` / `.pdf` / `.md`：独立讲稿。
- `待补图方法论文本.docx` / `.pdf` / `.md`：可选示意图需求。
- `slides.json` / `equations.json`：可重复生成的内容与公式。
- `figure_manifest.json` / `source_manifest.json`：原图和访问证据。
- `validation.json`：页数、讲稿、原图一致性与渲染检查。
- `SHA256SUMS`：文件校验。

重新生成：`python reports/build_task_aware_talk.py`。
依赖：python-pptx、python-docx、matplotlib、Pillow、PyMuPDF、soffice、Noto CJK字体。

分支：`slides/task-aware-paper-mit`。没有修改原训练/遗忘代码，没有重新训练或计算Cost，没有合并master。

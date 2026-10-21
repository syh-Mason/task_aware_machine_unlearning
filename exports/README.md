# 实验图表下载

## CNN 模型内部层作用路径

[下载层结构、激活、归因和内部消融完整包](cnn_layer_paths.zip?raw=true) ·
[七页Word/PDF与结果目录](../research/cnn_layer_paths/README.md)。

固定已有输入，分析 Conv1/ReLU/Conv2/Flatten/全连接/Tanh/预测头。
采用官方 Captum v0.7.0，完成四模型逐层记录、320次内部置零消融；不是温度输入扰动实验。
上传的 Captum ZIP 超过32 MiB读取上限，未读包内版本；使用的公开版本及解释边界已记录。

## TAMU 分层导图

[下载一张总图与八张详图完整包](tamu_layered_mindmaps.zip?raw=true) ·
[图集、Word 与交互版入口](../reports/tamu_layered_mindmaps/README.md)。

包含九页矢量导图PDF、高清PNG/SVG、十一页分层说明Word/PDF及可编辑导图。
按学习逻辑分层，非原文章节号；原文全文尚未取得，所有来源范围和扩展实验均已标注。

## TAMU 算法推导与先修文献

[下载 Word / PDF 讲义 ZIP](tamu_algorithm_guide.zip?raw=true) ·
[阅读说明与单独下载](../reports/tamu_algorithm_guide/README.md)。

包含先修文献链接、直接遗忘/PAMU/TAMU 数学推导、KKT/伴随求导、完整两阶段 Cost 与代码对应。
**这是代码核对版：当前代理阻止取得论文全文，尚未完成原文定理、公式编号的逐项核验。**
文档首页已披露来源限制，附 10 项教学数值检查，未新增或改写实验数据。

## 两次实验报告与组会汇报

[下载三份报告完整包 ZIP](cnn_experiment_reports.zip?raw=true) ·
[文档目录](../reports/cnn_experiment_reports/README.md)。

包含两份独立实验报告（Word/PDF），一份15页组会PPT及PDF，以及逐页讲稿与答疑备忘（Word/PDF）。
附实际数值表、图表和溯源记录，未重新训练或重新生成实验结果。

## CNN 输出节点 Cost 诊断与多日期验证

[下载本轮完整结果 ZIP](cnn_cost_attribution.zip?raw=true) ·
[导师汇报](../research/cnn_cost_attribution/导师汇报.md) ·
[完整报告](../research/cnn_cost_attribution/README.md)。

包含五档温度扰动的两阶段成本、四种冻结模型对照、输出节点隔离/复位与抽样 Shapley，
以及 27 个分层日期加原参考点的 28 日期实验。全部实际预测、成本、求解轨迹和复算脚本均保存。

## CNN 五档温度扰动与四模型对照

[下载本轮完整结果 ZIP](cnn_temperature_response.zip?raw=true) ·
[导师汇报](../research/cnn_temperature_response/导师汇报.md) ·
[完整实验报告](../research/cnn_temperature_response/README.md)。

固定一个测试样本和节点温度，按训练集标准差做 −1、−0.5、0、+0.5、+1σ 扰动；
对比原始、20% 直接遗忘、重训预测头和 20% TAMU（L1=0.1）。
包含三张响应图、一张流程图、完整逐节点预测/Cost 数据、冻结权重、真实输入与复算脚本。

## CNN 输入敏感性与 20% 遗忘研究

[下载五项任务完整结果 ZIP](cnn_sensitivity.zip?raw=true) ·
[查看在线研究报告](../research/cnn_sensitivity/README.md)。

包含原始输入140维和隐藏表示64维说明、CNN与cost流程图、逐坐标+10%扰动、遗忘前后敏感性对比、完整数值与可重放的真实输入/冻结模型快照。

## 三模型六比例图表

[下载完整图表 ZIP](experiment_figures.zip?raw=true)（约 1.58 MB）。

解压后，图表按 `linear`、`cnn`、`mlp` 三个文件夹分类，包含：

- 三套 MSE / MAPE / Cost 三联图及九张独立单图，均提供 PNG 和矢量 PDF。
- 六个遗忘比例（5%、10%、15%、20%、25%、30%）的实际实验数值 CSV / JSON。
- 复现报告、绘图脚本和文件校验清单。

也可直接浏览仓库中的 [图表文件夹](../figs/) 和 [图表复现报告](../FIGURE_REPRODUCTION_REPORT.md)。

下载后的 ZIP 校验值见 [SHA-256 文件](experiment_figures.zip.sha256)。压缩包保留导出时的实验报告；报告中关于未推送的说明记录的是此前本地实验阶段的状态。

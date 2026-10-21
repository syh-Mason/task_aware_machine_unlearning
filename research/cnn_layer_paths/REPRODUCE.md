# 复算说明

本次分析对象是原 Task-Aware CNN，不是 Captum 库本身的“网络层”。输入固定为此前保存的原始 test 0；没有进行温度输入扰动，没有训练或重新优化模型，也没有新增 Cost 计算。

用户提供的 `captum-master.zip` 超过文件工具 32 MiB 的读取上限，未能读取。实际使用的是官方公开 Captum **v0.7.0**，提交 `833d6aad27a921d2868ac0e248b4ab3ed50b13b9`，与既有 PyTorch 2.0.1+cpu 环境兼容；不声称等同上传 ZIP 的版本。

## 完整仓库中复算

在已有科学计算环境中，先准备独立的 Captum 源码目录，不改项目 `.venv`：

```bash
git clone --depth 1 --filter=blob:none --sparse --branch v0.7.0 \
  https://github.com/pytorch/captum.git \
  /workspace/.cloud-setup/task-aware/captum-v0.7.0
git -C /workspace/.cloud-setup/task-aware/captum-v0.7.0 sparse-checkout set captum
```

源码目录已存在时直接使用它。在本仓库根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/workspace/.cloud-setup/task-aware/captum-v0.7.0 \
.venv/bin/python experiments/analyze_cnn_layers.py \
  --source research/cnn_temperature_response \
  --output-dir /tmp/cnn-layer-replay
```

读取既有 `frozen_models.npz`、`reference_inputs.npz` 与此前逐节点预测 CSV。解压包内脚本是代码副本；上述复算命令需要完整仓库及已有项目依赖。

文档与图表使用系统文档环境重建：

```bash
python reports/build_cnn_layer_paths.py
```

需要 NumPy、Pandas、Matplotlib、Pillow、PyMuPDF、python-docx、Graphviz、LibreOffice 与 Noto CJK 字体。

## 结果怎样读

- `layer_statistics.csv`：逐层形状、参数数量、激活统计；不是层重要性的全局排名。
- `layer_attributions.csv`：每个输出在每层的 Gradient×Activation；卷积层按通道汇总，CSV同时保留带符号和绝对质量。
- `internal_ablations.csv`：固定原始输入，分别将两次 ReLU 的整通道或 Tanh 单个神经元置零后的预测变化。
- `layer_arrays.npz`：完整激活与归因数组、隐藏到输出的精确仿射贡献及常数项。
- `models.npz`、`fixed_input.npz`：本轮使用的精简模型和固定样本快照。
- `validation.json`：90项本轮内部一致性检查及方法、版本、来源限制。它不是此前实验的累计测试报告。

同一模型的不同层分数不能相加；整通道消融与单神经元消融的干预大小不同。只有最终仿射头的隐藏贡献在常数项单列后具有本轮验证的精确可加表达。所有结论仅对应此样本与所选归因/置零方法，没有证明全局因果或枚举最重要完整神经元路径。

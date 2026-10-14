# Cost 计算修复与实验复现报告

工作分支：`fix/cost-evaluation`。原始代码提交：`98a3e7bf177fe40d1d538e63a869578338d70c81`。未修改或合并到 master，未推送远程。

已修复并实际验证 cost 计算路径。**TX-123BT 实验尚未复现**：真实数据及对应 influence、真实训练检查点均不存在，实际下载域名仍被运行时网络规则拒绝。本报告中的数值来自明确标记的 PYPOWER case14 求解器算例，不能用作论文实验结果。此前环境初始化生成的合成数据保留在仓库外，本次没有将它们复制成真实数据，也没有运行合成数据来替代真实实验。

## 环境与安装

复用了此前建立的隔离环境 `.venv`，并按本次新增的 `requirements.txt` 同步、检查依赖。完整文件锁定了实际安装的 76 个包；不修改系统 Python。没有 GPU 或外部服务要求。

| 组件 | 实际版本 |
| --- | --- |
| Python | 3.10.22 |
| PyTorch | 2.0.1+cpu |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| pandas | 1.5.3 |
| CVXPY / CVXPYLayers | 1.2.3 / 0.1.5（发行包元数据） |
| diffcp | 1.0.23 |
| ECOS / OSQP / SCS | 2.0.12 / 0.6.3 / 3.2.3 |
| Gurobi | 13.0.2 |
| Hydra | 1.3.2 |
| openpyxl / xlrd | 3.1.2 / 2.0.1 |
| pytest | 8.4.2 |

README 指定 Gurobi 11.0.0；此前实际安装成功，但运行报 `GurobiError 10009: License expired 2025-11-24`，11.0.3 仍有相同限制。13.0.2 的内置非生产许可证当前有效至 2027-11-29，但有规模限制。本次另用真实 Gurobi API 建立 2100 变量的小模型，确认报 `10010 Model too large for size-limited license`。这与包安装失败不同。

初始化时 uv 默认缓存和可执行链接目录只读；已将缓存、Python 安装目录改为 `/workspace/.cache/uv`、`/workspace/.python`，并用 `--no-bin` 避免向只读用户目录写入链接。本次依赖同步无安装失败，`uv pip check` 通过。保留了 TLS、包校验，不使用关闭验证的参数。

安装与检查命令（仓库目录）：

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.python
uv python install --no-bin 3.10.22
# 仅在 .venv 尚不存在时创建，保留已有环境。
if [ ! -x .venv/bin/python ]; then
  uv venv --python /workspace/.python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 .venv
fi
uv pip sync --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cpu \
  --index https://pypi.org/simple --index-strategy unsafe-best-match requirements.txt
uv pip check --python .venv/bin/python
```

完整可重复安装脚本位于 `/workspace/.cloud-setup/task-aware/install-cost.sh`，已实际执行成功。安装不生成实验数据。

## 数据核查

读取了 README、全部配置和 cost 依赖路径上的源码，包括数据预处理、训练、影响函数、两阶段优化、评估和 notebook 结果读取。原仓库没有 requirements 文件，也没有 `run_real.py`；本次新增了这两个文件。

README 需要 `data/raw_data.zip`，解压后应有 `data/Data_public/`。预处理读取 2019 年：365 个 `load_2019/load_annual_D{day}.txt`，以及 365 个 `Climate_2019/climate_2019_Day{day}.csv`。后者虽然名为 CSV，代码用 Excel 工作簿读取，每天包含 `Hour 1` 至 `Hour 24`。预处理最终应生成 14 个 `bus_*.csv`，每个 8760 行、10 个特征和 `Load` 列。

当前核查：

| 输入 | 当前状态 |
| --- | --- |
| 原始归档 | 不存在 |
| 上述 730 个原始输入文件 | 0 个 |
| `data/data_case14/bus_*.csv` | 0 个 |
| `influence/linear_mse.npy`、其他真实数据 scores | 不存在 |
| `trained_model` 真实模型检查点 | 仅有 `.placeholder` |

已实际访问官方 Figshare API，确认项目官网链接对应文章 `22144616` 的**版本 4**：

- 文件 ID：`39478540`；文件名：`Texas Synthetic Power System Test Case (TX-123BT).zip`。
- 元数据记录大小：`137051180` 字节。
- 官方 `computed_md5`：`fdd7bba5098f1071cbe1968a99587085`。
- 元数据：https://api.figshare.com/v2/articles/22144616/versions/4
- 实际下载：https://ndownloader.figshare.com/files/39478540

上述大小和 MD5 是官方元数据，**本地归档尚未下载，因此尚未完成本地校验**。文章最新版本 6 是五年数据，不能直接当成 README 要求的 2019 原始目录。

`api.figshare.com` 已可访问。最初 `ndownloader.figshare.com` 报 `CONNECT tunnel failed, response 403`，追加该域名后，后续实际请求已返回 302，确认文件存储在 `s3-eu-west-1.amazonaws.com`。跟随实际下载仍被该存储域名的 CONNECT 403 阻断，未取得归档。已将这个经实际响应确认的存储域名继续追加到环境配置草稿，保留现有域名及包管理预设。保存草稿不会自动改变运行时网络，需要在环境设置保存配置使其生效，之后重新验证完整下载。公开数据无需新增密钥。

## 原始故障与修复

| 原始表现 | 根本原因 | 修复 |
| --- | --- | --- |
| `python run_real.py`: 文件不存在 | 原仓库没有该入口 | 新增只接受真实数据的运行入口，缺数据时非零退出 |
| `FileNotFoundError: influence/linear_mse.npy` | 未先运行 `gen_index.py`，或生成/读取路径、数据规模不一致 | 在最终训练/敏感数据集建立后读取 scores，校验路径、长度、一维形状和有限值；运行入口统一生成/读取路径 |
| 两条样本只返回一个 cost，实测 `2874.35862587` | `len(pred)==2` 把两行 Tensor 当成 `(features, prediction)`；再调度通过 `zip` 截断批次 | 按 tuple/list 类型识别 NN 输出；预测与目标形状必须一致；禁止批次截断 |
| 相同两条样本单独评估是约 `119.90729842`、`294.13454889` | 上述批量错误改变了预测输入、丢失样本 | 回归测试要求批量 cost 等于逐条 cost |
| 可微 cost 在过载算例约为 `84832.6328125`，直接求解约 `137343.689558` | 可微路径仅返回阶段一发电量，漏掉阶段一负荷削减费用；默认 SCS 精度还造成小量数值误差 | 保留阶段一削减量，并把已有目标函数中的该费用加回 `cost1+cost2`；提高层求解精度 |
| `RuntimeError: expected scalar type Double but found Float` | cost 系数固定 float32，与 float64 求解输出不匹配 | 发电 cost 系数匹配输出 dtype/device |
| 无法使用 Gurobi、规模超限后直接终止 | 三处强制 GUROBI，未提供等价兼容求解器 | 同一个 CVXPY problem 自动尝试 GUROBI→ECOS→OSQP→SCS；显式指定求解器时不偷偷换用其他求解器 |
| 失败求解可能继续读取 None 或旧值 | 先读变量、未验证求解状态、有限值与约束残差 | 只接受 `optimal`、有限目标/变量且最大约束违反不超过 `1e-5` 的结果；否则明确失败或在自动模式尝试兼容求解器 |
| 对 CVXPY 的变量/参数内部顺序有依赖 | 按 `.variables()[0]` 等位置取值，层参数依赖遍历顺序 | 使用语义命名、显式变量引用和显式层参数映射 |
| `eval_unchange.py` 单独运行时无法保存结果 | 未创建 `simulation_result/<model>` | 进入运行时创建该输出目录 |
| 第一个气候 bus 取不到数据，其他 bus 错位 | 相关矩阵/负荷列是零起始索引，却用 `iloc[bus-1:bus]` | 使用 `iloc[bus:bus+1]`，索引单元测试验证；真实预处理仍未运行 |

修改文件：

- `utils/optimization.py`：兼容求解器、状态/残差检查、显式映射、输入及批次验证。
- `utils/funcs.py`：两条样本误判修复、预测形状检查、influence 输入校验。
- `utils/net.py`：明确层变量/参数、保留阶段一削减量、与评估路径一致的负预测截断、SCS 精度。
- `func_operation.py`：完整两阶段可微 cost、动态维度拆分、dtype 匹配。
- `eval_unlearn.py`、`eval_unchange.py`：一致的 scores 校验；后者还使用兼容求解器、准备输出目录。
- `clean_data.py`：气候行与负荷列索引对齐。
- 新增 `requirements.txt`、`run_real.py`、`scripts/cost_smoke.py`、`tests/test_cost.py` 和本报告。

## 完整计算路径与数学保持

`clean_data.py` → `MyDataset` 根据 case14 标准负荷缩放为 p.u. 并标准化 → 解析线性拟合或 NN 核心训练/最后层拟合 → `gen_index.py` 生成影响分数 → `eval_unlearn.py` 选样本、遗忘/重训练 → `utils.funcs.evaluate` 还原目标尺度 → `Operator.solve_one(pred)` → `Operator.solve_two(target, pg)` → 对 `cost1+cost2` 取均值或逐样本输出。

第一阶段仍最小化：

`pg^T diag(second_coeff) pg + first_coeff^T pg + load_shed_coeff * sum(ls1)`。

第二阶段仍最小化：

`load_shed_coeff * sum(ls2) + gen_storage_coeff * sum(gs) + load_shed_coeff_second * sum(ls2^2) + gen_storage_coeff_second * sum(gs^2)`。

保持原有发电上下限、线路双向限额、两阶段功率平衡、削减/储存非负和参考节点角约束。不增加 `ls<=load`、`gs<=pg` 等原项目没有的约束，不替换为其他 cost 定义，不修改系数、目标尺度、采样/遗忘算法或训练损失。

已实际比较原始提交与修改后源码的 AST，两个 `stage_one` / `stage_two` 的目标函数与全部约束赋值完全一致。ECOS/SCS 使用 CVXPY 对凸二次目标的等价锥化（正半定二次项的 epigraph），OSQP 直接求解同一凸 QP；没有手工改写优化模型。除目标值对照，测试还分别检查了原始目标公式、功率平衡、线路限额、发电边界和参考角。

可微路径是 `SPO` → 两个 CVXPYLayers → `MSE_COST.test_loss` → 测试梯度 → iHVP → 重加权。原来的内部输出 `[forecast, pg, ls2, gs]` 现在追加 `ls1`，从已有第一阶段目标恢复削减费用；`MSE_COST` 同时更新维度拆分。case14 的分块维度为 `[14,5,14,5,14]`。NN 预测器架构和训练检查点格式不变。外部代码若直接消费 SPO 拼接输出，需要采用这个布局。

`TAMU_SOLVER` 控制普通调度、再调度和重加权问题；CVXPYLayers 的可微求解仍使用 diffcp/SCS，设 `eps=1e-7, max_iters=100000`。它们求解同一数学模型，但有限精度产生的小差异应如实保留。旧的 cost influence 文件在该修复后应重新生成；修改天气索引后，也应重新预处理并重新训练，保留旧结果备查。

## 实际运行与数值

最小命令已实际执行：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest tests -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.cost_smoke
.venv/bin/python run_real.py --check-data
.venv/bin/python run_real.py --data-prop 0.01 --solver ECOS
HYDRA_FULL_ERROR=1 .venv/bin/python eval_unlearn.py model=linear
```

自动化测试：**23 passed，0 failed，0 skipped**。覆盖批量 cost、直接/可微 cost 一致性、负预测截断、float64 梯度、四个求解器、Gurobi 缺失/许可证失败后等价求解、超规模重加权 LP、不可行时拒绝旧值、无效输入、scores、气候索引和缺数据时不制造输出。存在 Hydra `_self_`、diffcp/distutils 的已知警告；超规模 Gurobi 自动切到 ECOS 的警告是实际执行的降级记录，没有被当成 Gurobi 成功。

以下是 **PYPOWER case14 的真实求解器计算值**：预测负荷为配置默认值的 `0.7` 倍，真实负荷为 `0.714` 倍。不是 TX-123BT 训练/测试集结果。

| 求解器 | cost1 | cost2 | total cost | 两阶段状态 |
| --- | ---: | ---: | ---: | --- |
| GUROBI | 199.5343811283 | 372.2421475783 | 571.7765287066 | optimal / optimal |
| ECOS | 199.5343809973 | 372.2421531933 | 571.7765341906 | optimal / optimal |
| OSQP | 199.5343810390 | 372.2421505892 | 571.7765316281 | optimal / optimal |
| SCS | 199.5343440137 | 372.2421201172 | 571.7764641309 | optimal / optimal |

过载回归算例（预测、真实负荷均为配置默认值的 3 倍）：

- 第一阶段削减费用：`52508.18322295437`。
- ECOS 直接两阶段 cost：`137343.68992576541`。
- 修正后可微 cost：`137343.68821003268`。
- 差值约 `0.001716`，相对误差约 `1.25e-8`；反向传播梯度有限且非零。

实际数值 JSON：`outputs/cost-debug/cost-benchmarks.json`。这份文件明确标注非 TX-123BT 数据来源。

真实运行入口的两个命令均退出码 1，审计显示缺少全部真实输入，没有生成真实实验或 cost 值。直接 `eval_unlearn.py` 也退出码 1，现明确先报 `FileNotFoundError: data/data_case14/`；旧代码首先报缺少 influence。真实 `gen_index.py`、`eval_unchange.py`、CNN/MLP-Mixer 的完整训练及论文图表未运行，不记为通过。本次沒有通过跳过 cost 来让真实流程成功。

实际日志和版本清单位于 `/workspace/.cloud-setup/task-aware/debug-logs/`：`before-tests.log`、`final-tests.log`、`install.log`、`reusable-install.log`、`versions.json`、`cost-benchmarks.log`、`real-data-audit.log`、`real-run.log`、`real-eval-unlearn.log`。官方元数据留存于 `/workspace/.cloud-setup/task-aware/real-logs/dataset-v4.json`。

## 恢复真实实验的步骤与限制

在环境设置保存新增的 `s3-eu-west-1.amazonaws.com` 网络配置，保留已恢复访问的 `ndownloader.figshare.com`；重新测试下载，获取版本 4 归档并验证上述大小/MD5。若允许发布当前环境，可随后发布保存的开发环境；本次没有执行发布。解压时保留已有用户文件，确认上述 2019 目录和真实气候文件格式。当前数据目录为空，尚无旧数据被覆盖。

有完整真实数据后，从仓库执行：

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python run_real.py --check-data
python run_real.py --data-prop 0.01 --solver ECOS --output-dir outputs/real-small
# 小规模通过后，使用全部 8760 小时和原始默认的所有权重约束：
python run_real.py --data-prop 1 --solver ECOS --output-dir outputs/real-full
```

入口按顺序执行必要的真实预处理、`check.py`、`gen_index.py`、`eval_unlearn.py`、`eval_unchange.py criteria=cost`，不跳过 cost。对应 influence 与 simulation 路径位于指定输出目录，数据比例、遗忘比例保持一致。默认保留原项目的线性 `linear_mse.npy` 选样分数，而不擅自改成按 cost 选样。

也可运行原 README 命令：`python clean_data.py`、`python check.py model=linear unlearn_prop=0.2`、`python gen_index.py model=linear`、`TAMU_SOLVER=ECOS python eval_unlearn.py model=linear unlearn_prop=0.2`、`TAMU_SOLVER=ECOS python eval_unchange.py model=linear criteria=cost unlearn_prop=0.2`。生成与读取 scores 必须采用相同模型、数据比例、随机种子等设置；长度校验不能证明同长度 scores 的完整来源，因此不要复用未知来源的旧 scores。

NN 实验还需要用真实数据执行 `python train_nn.py model=conv` 或 `model=mlpmixer`（README 默认 100 epochs）后，再生成对应 scores 并评估。未验证全数据性能、神经网络完整实验或论文数值。case9 配置还缺少若干原代码必需字段，本次只修复和验证 README 默认的 case14，不宣称 case9 可以运行。

当前的必要外部动作是恢复真实数据下载访问，或者提供原始真实归档。兼容求解器已验证，不再必须靠新增 Gurobi 许可证才能计算 cost；若明确强制 GUROBI，仍须满足其许可证期限和规模限制。

环境配置草稿已保存并确认：`install_script` 是测试通过的完整安装脚本，`start_skill` 包含激活、数值检查、真实数据运行步骤和限制；网络列表追加下载域名并保留已有设置。这些保存不执行发布，当前机器验证不等于新任务验证。修复分支未推送，后续任务的本地分支/提交恢复也未独立验证。

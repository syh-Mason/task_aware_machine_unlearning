# Cost 修复与实验复现报告

已用官方 TX-123BT 完整年度数据完成 case14 的线性、CNN、MLP-Mixer 实验：遗忘比例 20%，三种模型均执行影响分数生成、基线遗忘、重训练对照、任务感知 cost 优化。共完成 25 组权重约束（线性 11、CNN 7、MLP-Mixer 7）。26 项自动化测试通过。

后续已完成三个模型在 5%、10%、15%、20%、25%、30% 的 18 组基线实验，生成每个模型的 MSE / MAPE / Cost 三联图及单图。完整数值、图表与复现命令见 [图表复现报告](FIGURE_REPRODUCTION_REPORT.md)。

代码位于独立分支 fix/cost-evaluation，基于原提交 98a3e7bf177fe40d1d538e63a869578338d70c81。未合并 master、未推送远程、未执行环境发布。

## 实际结果

以下为完整官方数据上的测试集平均 cost，沿用原仓库单位和 cost1 + cost2 定义。TAMU 固定展示平均 L1 权重约束 0.1，没有按最佳结果挑选约束。

| 模型 | 原始模型 | 基线遗忘 | 重训练 | TAMU，L1=0.1 |
| --- | ---: | ---: | ---: | ---: |
| linear | 928.603650 | 1061.601844 | 1061.601843 | 836.535090 |
| nn_conv | 476.944457 | 546.221348 | 546.221062 | 470.887519 |
| nn_mixer | 511.728601 | 560.110506 | 560.111122 | 489.875626 |

使用配置：case14、data_prop=1、train_prop=0.8、random_seed=42、unlearn_prop=0.2、unlearn_mode=helpful。线性模型按原默认 MSE influence 选样，两个 NN 按原默认 cost influence 选样。

- 2019 年共 8760 小时。训练集 7008、测试集 1752。
- NN 核心集 2102，敏感训练集 4906；两个核心网络均训练 100 epochs。
- 线性删除 1401 条、保留 5607 条；NN 删除 981 条、保留 3925 条。
- 完整曲线数据：[cost-curves.csv](outputs/cost-debug/cost-curves.csv)。
- 实际结果和配置摘要：[real-results.json](outputs/cost-debug/real-results.json)。
- 曲线图：[PNG](outputs/cost-debug/cost-curves.png)、[PDF](outputs/cost-debug/cost-curves.pdf)。

![完整官方数据的 cost 曲线](outputs/cost-debug/cost-curves.png)

结果文件、每种 influence 的样本数量、所有 MSE/MAPE/cost 的有限性，以及 25 组约束的完成情况均已核验。最终结果的时间晚于对应 influence 生成日志，不是旧输出或零测试结果。

## 数据来源与完整性

README 要求的归档来自官方 Figshare 文章 22144616 的版本 4，文件 ID 39478540，而不是最新版五年归档。

- 元数据：https://api.figshare.com/v2/articles/22144616/versions/4
- 下载：https://ndownloader.figshare.com/files/39478540
- 文件名：Texas Synthetic Power System Test Case (TX-123BT).zip
- 本地字节数与官方一致：137051180。
- 本地 MD5 与官方 computed_md5 一致：fdd7bba5098f1071cbe1968a99587085。
- ZIP 共 4252 个成员，所有 CRC 校验通过。
- 项目使用的 365 个 load_2019 文本和 365 个 Climate_2019 工作簿全部存在。
- 气候文件虽命名为 .csv，实际按 Excel 读取；每日包含 Hour 1 至 Hour 24，小时表为 123 行、7 列，Bus 编号从 1 开始。
- 预处理实际生成 14 个 bus CSV，每个 8760 行、11 列，全部有限值，负荷有效且非常数。

下载早期经历代理 CONNECT 403。根据真实响应依次追加 ndownloader.figshare.com 和 s3-eu-west-1.amazonaws.com，随后重新请求成功。没有更换预计校验和或关闭 TLS 验证。归档位于 data/raw_data.zip，原始文件完整保留，派生数据位于 data/data_case14。官方数据本身是公开的合成 Texas 系统数据，本次没有自行编造缺失数据。此前环境初始化的测试合成数据仍在仓库外，未用于这些实验。

## 环境与安装记录

使用隔离环境 .venv，不修改系统 Python。新增 requirements.txt，锁定实际安装的全部 76 个包。

| 组件 | 实际版本 |
| --- | --- |
| Python | 3.10.22 |
| torch | 2.0.1+cpu |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| pandas | 1.5.3 |
| CVXPY / CVXPYLayers | 1.2.3 / 0.1.5（发行包元数据） |
| diffcp | 1.0.23 |
| ECOS / OSQP / SCS | 2.0.12 / 0.6.3 / 3.2.3 |
| gurobipy | 13.0.2 |
| Hydra | 1.3.2 |
| openpyxl / xlrd | 3.1.2 / 2.0.1 |
| pytest | 8.4.2 |

README 的 Gurobi 11.0.0 安装成功，但运行报 10009: License expired 2025-11-24；11.0.3 同样过期。13.0.2 小规模 QP 可用，内置非生产许可证有效至 2027-11-29，但规模受限。另用实际 Gurobi API 建立 2100 变量模型，确认 10010: Model too large for size-limited license。完整重加权问题由自动模式切到 ECOS，未将 Gurobi 的失败记为成功。

uv 默认缓存、可执行链接目录只读，已改用 /workspace/.cache/uv、/workspace/.python，并指定 --no-bin。本次冻结依赖同步没有安装失败，uv pip check 通过。安装日志和完整版本清单位于 /workspace/.cloud-setup/task-aware/debug-logs/。可重复安装脚本 /workspace/.cloud-setup/task-aware/install-cost.sh 已实际执行；不会生成实验数据。

## 原始报错、根本原因与修改

| 报错或错误结果 | 原因 | 最小修复 |
| --- | --- | --- |
| run_real.py 文件不存在 | 原仓库没有该入口或 requirements | 新增真实数据审计、顺序执行原命令的入口及冻结依赖文件 |
| 缺少 influence/linear_mse.npy | 未生成 scores，或生成与读取路径/数据规模不一致 | 在最终训练/敏感集建立后读取；验证路径、样本数、一维形状、有限值；入口统一路径 |
| 两条样本只返回一个错误 cost：2874.358626 | len(pred)==2 把两行 Tensor 当作 NN 的返回 tuple；zip 再调度静默丢弃样本 | 按返回类型判断；预测和目标形状必须一致；批次大小不一致时报错 |
| 同两条样本单独 cost 为 119.907298、294.134549 | 上述批量误判 | 自动化测试要求批量与逐条结果一致 |
| 过载可微 cost 约 84832.6328，直接 cost 约 137343.6896 | 可微路径丢弃阶段一削减量，遗漏阶段一削减费用约 52508.18 | 保留阶段一 ls1，并从原有第一阶段目标恢复该费用；与 cost1+cost2 一致 |
| Double/Float 类型不匹配 | cost 系数固定 float32，两个优化层参数也可能混合 dtype | 系数匹配输出类型；目标匹配可微层参数类型 |
| Gurobi 过期或规模超限直接失败 | 三处强制使用 GUROBI | 同一个 CVXPY problem 自动尝试 GUROBI→ECOS→OSQP→SCS；显式指定时不偷偷切换 |
| 真实 ECOS 再调度返回 optimal_inaccurate | 某条实际输入无法达到先前设置的 1e-9 容差，实际残差约 4.25e-8，非模型不可行 | 用记录的真实输入比较四种求解器，ECOS 使用 1e-8；严格状态检查保留，自动模式在其他不准确样本上改用兼容求解器 |
| CG info=1000，但原检查仍退出 0 | 解析拟合的 double 系数被截成 float32；逆 Hessian 计算达到迭代上限却照常返回 | 保留仿射模型双精度、匹配 MSE 目标类型、重构时保留 dtype；未收敛时明确失败 |
| 依赖 CVXPY 变量/参数遍历顺序 | 按变量位置取 PG、LS 等 | 使用语义命名和显式引用、显式层参数映射 |
| 失败求解仍可能读 None 或旧值 | 未检查求解状态/有限值/残差 | 只接受 optimal、有限目标/变量且约束违反不超过 1e-5 的结果，复制当前解后返回 |
| 气候 Bus 0 行为空、其他 Bus 错位 | 负荷/相关矩阵索引从 0 开始，却使用 iloc[bus-1:bus] | 使用 iloc[bus:bus+1]，由真实工作簿的 Bus 编号及单元测试验证 |
| 独立运行 eval_unchange 无输出目录 | 未创建 simulation_result/model | 在入口创建目录 |
| simulation_dir 无结尾斜杠时 baseline 保存到 simulation_resultlinear 等目录 | eval_unlearn 使用字符串拼接 | 改为 Path 拼接；小规模真实评估验证新路径。完整 baseline 的已计算文件保留原件、逐字节一致复制到规范目录，未重写数值 |

修改文件：utils/optimization.py、utils/funcs.py、utils/net.py、func_operation.py、torch_influence/modules.py、clean_data.py、eval_unlearn.py、eval_unchange.py；新增 requirements.txt、run_real.py、scripts/cost_smoke.py、tests/test_cost.py、tests/fixtures/redispatch_ecos.json 和本报告。fixture 是实际失败再调度输入，不是制造的实验数据。

## 数学模型与完整计算路径

clean_data → MyDataset 负荷尺度与标准化 → 解析线性拟合或真实 NN 核心训练/敏感集最后层拟合 → gen_index 生成影响分数 → 选样、遗忘/重训练 → evaluate 还原尺度 → solve_one(pred) → solve_two(target, pg) → 平均 cost1 + cost2。

第一阶段仍为：

pgᵀ diag(second_coeff) pg + first_coeffᵀ pg + load_shed_coeff * sum(ls1)。

第二阶段仍为：

load_shed_coeff * sum(ls2) + gen_storage_coeff * sum(gs) + load_shed_coeff_second * sum(ls2²) + gen_storage_coeff_second * sum(gs²)。

保留原发电上下限、线路双向限额、两阶段功率平衡、削减/储存非负、参考节点角约束。不增加 ls<=load 或 gs<=pg 等原项目没有的约束。没有增加岭回归、改变 damp、修改训练损失或原始 CG 相对容差 1e-5。

已比较原始提交与修复后 stage_one/stage_two 的 AST：目标函数和所有约束赋值完全一致。ECOS/SCS 使用 CVXPY 对正半定二次项的等价锥化，OSQP 求解同一凸 QP；只更换求解算法，没有手工改变优化问题。标准 case14 上四个求解器的总 cost 约 571.77653，目标值和物理约束分别通过测试。

默认 case14 的两阶段可行性也可构造验证：阶段一取 PG=0、LS1=预测负荷、角度=0；阶段二取 LS2=真实负荷、GS=已调度 PG、角度=0，满足原约束。非负输入下的 ECOS 不准确状态不能被当成物理不可行。

可微路径为 SPO → 两个 CVXPYLayers → MSE_COST → 测试梯度 → iHVP → 原重加权 LP。SPO 内部输出从 [forecast, PG, LS2, GS] 追加 LS1，case14 的分块长度为 [14,5,14,5,14]。预测器架构和核心训练检查点不变；仿射拟合/重构保留 float64。负预测截断与原 evaluate 的输入处理一致。可微层使用 diffcp/SCS，eps=1e-7、max_iters=100000；普通求解及重加权由 TAMU_SOLVER 控制。

## 实际命令与检查

全部命令从仓库目录执行，使用 .venv/bin/python。每个已有脚本记录退出码，最终完整流水线全部退出 0。

~~~bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest tests -q
.venv/bin/python -m scripts.cost_smoke
.venv/bin/python run_real.py --check-data
.venv/bin/python run_real.py --data-prop 0.01 --solver auto --output-dir outputs/final-smoke
.venv/bin/python run_real.py --data-prop 1 --solver auto --output-dir outputs/real-full-double
.venv/bin/python train_nn.py model=conv
.venv/bin/python train_nn.py model=mlpmixer
.venv/bin/python check.py model=conv unlearn_prop=0.2
.venv/bin/python check.py model=mlpmixer unlearn_prop=0.2
~~~

两个 NN 的完整命令分别使用 model=conv / model=mlpmixer，并使用原默认 cost influence。例如 CNN：

~~~bash
.venv/bin/python gen_index.py model=conv influence_dir=outputs/real-conv/influence
.venv/bin/python eval_unlearn.py model=conv model.influence_dir=outputs/real-conv/influence/nn_conv_cost.npy simulation_dir=outputs/real-conv/simulation_result
.venv/bin/python eval_unchange.py model=conv criteria=cost model.influence_dir=outputs/real-conv/influence/nn_conv_cost.npy simulation_dir=outputs/real-conv/simulation_result
~~~

MLP-Mixer 同样执行三个命令，将 model 改为 mlpmixer、目录改为 outputs/real-mixer、文件名改为 nn_mixer_cost.npy。

26 项自动化测试：0 failed，0 skipped。覆盖两条样本 cost、完整可微 cost、float64 梯度、四种求解器的原始目标公式与物理约束、失效/缺失 Gurobi、超规模 LP、失败后拒绝旧值、输入/批次维度、influence、气候索引、实际 ECOS 失败样本、解析拟合驻点以及 CG 非收敛处理。三个模型的已有 check.py 均通过，修正后所有 CG info 均为 0。两个核心网络均完成 100 epochs。

baseline 与重训练的测试 cost 差分别约 1.74e-7、2.87e-4、6.16e-4；这是原始 CG 容差下的数值近似，不宣称逐参数完全相同。

## 日志、保存配置与限制

最终完整输出位于 outputs/real-full-double、outputs/real-conv、outputs/real-mixer，包含对应 gen-index、eval-unlearn、eval-task-aware 日志与结果。环境、初始报错、失败样本、训练和检查日志位于 /workspace/.cloud-setup/task-aware/debug-logs。官方元数据位于 /workspace/.cloud-setup/task-aware/real-logs/dataset-v4.json。

此前小规模强制 ECOS 的不准确失败和旧 float32 CG 非收敛日志均保留；一次旧精度的完整运行在纠正 dtype 时停止并以新目录重跑，不将其未完成的输出计为成功。

配置草稿保存完整 install_script 和 start_skill，以及五个已确认需要的自定义网络域名；保留包管理预设。保存配置不执行环境发布，新任务的分支/提交恢复未独立验证。原始数据、检查点、数值结果保留在现有 ignored 路径，不提交大型数据到 Git。

当前默认 case14、20% 遗忘、三种模型的 cost 工作流没有未解决的必需环境阻断。后续六个遗忘比例的三模型基线及其三联图也已完成。仍有这些具体限制：强制 ECOS 在部分样本上可能返回 optimal_inaccurate，应使用自动模式；强制 Gurobi 受许可证期限/规模限制，自动模式已在实际重加权上使用 ECOS。未运行跨比例的完整 PUMU/TAMU 曲线、PUMU 的全部 mse/mape 曲线或完整 analysis.ipynb，也未逐图逐项对照原论文数值。case9 配置仍缺若干必要字段，本次不宣称 case9 可运行。

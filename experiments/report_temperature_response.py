"""Render exactly three measured temperature response figures and a short report."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.input_sensitivity import NAMES
from scripts.analyze_cnn_sensitivity import write_json

COLORS=['#2563a6','#dc5d37','#229762','#8e44ad']
STYLES=['-','--',':','-.']


def save(fig,directory,name):
    fig.savefig(directory/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(directory/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'research/cnn_temperature_response')
    args=parser.parse_args()
    out=args.results_dir.resolve()
    manifest=json.loads((out/'manifest.json').read_text())
    stats,metadata=manifest['temperature_statistics'],manifest['model_restore']
    frame=pd.read_csv(out/'results.csv')
    buses=pd.read_csv(out/'predictions_14_buses.csv')
    if len(frame)!=20 or set(frame.model)!=set(NAMES):
        raise ValueError('Figures require the completed four-model, five-level study')
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    original=frame[frame.model=='original'].sort_values('sigma_level')
    fig,ax=plt.subplots(figsize=(9.2,5.1),layout='constrained')
    palette=plt.get_cmap('tab20')
    for bus in range(1,15):
        subset=buses[(buses.model=='original')&(buses.grid_bus==bus)].sort_values('sigma_level')
        ax.plot(subset.sigma_level,subset.delta_mw,'o-',color=palette(bus-1),linewidth=1.5,
                markersize=4,label=f'Output bus {bus:02d}')
    ax.axhline(0,color='0.5',linewidth=.7)
    ax.set(xlabel='Temperature perturbation (training standard deviations)',
           ylabel='Change in predicted load (MW)',xticks=[-1,-.5,0,.5,1],
           title=f'Original CNN: change temperature at input bus {stats["grid_bus"]}; fixed test sample {metadata["reference_test_index"]}')
    ax.legend(ncol=2,fontsize=8,loc='center left',bbox_to_anchor=(1.01,.5))
    ax.grid(alpha=.22)
    save(fig,figures,'01_temperature_prediction')
    fig,ax=plt.subplots(figsize=(8.8,4.7),layout='constrained')
    for column,label,color in [('cost1','Stage 1 cost','#2563a6'),('cost2','Stage 2 cost','#dc5d37'),
                                ('cost_total','Total cost','#229762')]:
        ax.plot(original.sigma_level,original[column],'o-',label=label,color=color,linewidth=1.8)
    ax.set(xlabel='Temperature perturbation (training standard deviations)',
           ylabel='Cost (original repository objective units)',xticks=[-1,-.5,0,.5,1],
           title='Original CNN: forecast enters both unchanged optimization stages')
    ax.grid(alpha=.22);ax.legend()
    save(fig,figures,'02_temperature_cost')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for index,(name,label) in enumerate(NAMES.items()):
        rows=frame[frame.model==name].sort_values('sigma_level')
        for ax,column,ylabel in zip(axes,['forecast_total_delta_mw','cost_delta','cost_total'],
                ['Change in total predicted load (MW)','Change in total cost','Total cost']):
            ax.plot(rows.sigma_level,rows[column],marker=['o','s','x','D'][index],
                    linestyle=STYLES[index],color=COLORS[index],label=label,linewidth=1.7,markersize=5)
            ax.set(xlabel='Temperature perturbation (training SD)',ylabel=ylabel,xticks=[-1,-.5,0,.5,1])
            ax.grid(alpha=.22)
    axes[0].set_title('Forecast response to temperature')
    axes[1].set_title('Cost response relative to own baseline')
    axes[2].set_title('Absolute cost with the same actual load')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('Four frozen heads, one shared CNN core; direct unlearning and retraining nearly overlap')
    save(fig,figures,'03_four_model_comparison')
    dot='''digraph Path {
      graph [rankdir=TB,bgcolor="white",fontname="DejaVu Sans",pad="0.2"];
      node [shape=box,style="rounded,filled",fillcolor="#edf4fb",fontname="DejaVu Sans",fontsize=11];
      raw [label="Official TX-123BT / case14\n14 CSVs x 10 features; feature 4 (zero-based) = Temperature"];
      stats [label="Original 7008 training rows\nCompute selected temperature SD (ddof=0)\nNo refitting of the original annual normalization"];
      input [label="Fixed test sample: shape (1,1,14,10)\nOne temperature += {-1,-0.5,0,0.5,1} x training SD"];
      core [label="Frozen CNN core\nConv1 -> ReLU -> Conv2 -> ReLU -> Flatten(280) -> Linear(64) -> tanh"];
      heads [label="Four affine heads: original / direct / retrained / TAMU\n64 learned features -> 14 standardized load predictions"];
      unscale [label="forecast_pu = output_z * target_std + target_mean\nMW = p.u. x baseMVA (100)"];
      one [label="Original Operator.solve_one(forecast)\nDispatch PG, LS1, angles; objective = cost1"];
      two [label="Original Operator.solve_two(fixed actual load, PG)\nRedispatch LS2, GS, angles; objective = cost2"];
      result [label="14 bus predictions / MSE / MAPE\nTotal Cost = cost1 + cost2; solver and constraint audits"];
      raw -> input; stats -> input; input -> core -> heads -> unscale -> one -> two -> result;
    }'''
    dotfile=figures/'input_to_cost.dot';dotfile.write_text(dot,encoding='utf-8')
    for extension in ['png','pdf','svg']:
        subprocess.run(['dot',f'-T{extension}',str(dotfile),'-o',str(figures/f'input_to_cost.{extension}')],check=True)
    report(out,manifest,frame,buses)
    print(f'Saved three response figures, one flow diagram and report: {out}',flush=True)


def report(out,manifest,frame,buses):
    s,m=manifest['temperature_statistics'],manifest['model_restore']
    rows=frame[frame.model=='original'].sort_values('sigma_level')
    original_table='\n'.join(f'| {r.sigma_level:+.1f} | {r.forecast_total_delta_mw:+.6f} | {r.mse_pu2:.9f} | {r.mape_percent:.6f} | {r.cost1:.6f} | {r.cost2:.6f} | {r.cost_total:.6f} |' for r in rows.itertuples())
    model_table=[];summaries={}
    for name in NAMES:
        values=frame[frame.model==name].set_index('sigma_level')
        base=values.loc[0.];pos=values.loc[1.];neg=values.loc[-1.]
        affected=buses[(buses.model==name)&(buses.sigma_level==1.)]
        changed=affected.loc[affected.delta_mw.abs()>1e-6,'grid_bus'].astype(int).tolist()
        max_bus=affected.loc[affected.delta_mw.abs().idxmax()]
        summaries[name]=dict(baseline_cost=float(base.cost_total),
            plus1_total_prediction_delta_mw=float(pos.forecast_total_delta_mw),
            plus1_cost_delta=float(pos.cost_delta),minus1_cost_delta=float(neg.cost_delta),
            plus1_changed_output_buses_above_1e_6_mw=changed,
            plus1_largest_response_bus=int(max_bus.grid_bus),plus1_largest_response_mw=float(max_bus.delta_mw),
            central_difference_total_prediction_mw_per_sigma=float((pos.forecast_total_delta_mw-neg.forecast_total_delta_mw)/2),
            central_difference_cost_per_sigma=float((pos.cost_delta-neg.cost_delta)/2))
        model_table.append(f'| {NAMES[name]} | {base.cost_total:.6f} | {pos.forecast_total_delta_mw:+.6f} | {pos.cost_delta:+.6f} | {neg.cost_delta:+.6f} |')
    write_json(out/'summary.json',summaries)
    d=frame[frame.model=='unlearned20'].sort_values('sigma_level')
    r=frame[frame.model=='retrained'].sort_values('sigma_level')
    direct_gap=float(np.max(np.abs(d.cost_total.to_numpy()-r.cost_total.to_numpy())))
    plus_original=frame[(frame.model=='original')&(frame.sigma_level==1.)].iloc[0]
    core_scope='沿用原项目的实验协议：卷积核心固定，仅拟合/遗忘最后 910 个仿射参数。这里“重新训练 CNN”指在剩余敏感样本上重新拟合预测头，没有从头重训全部卷积层。'
    tamu_vs_direct=summaries['tamu20']['plus1_cost_delta']-summaries['unlearned20']['plus1_cost_delta']
    text=f'''# CNN 单节点温度扰动：导师汇报

已完成截图中的 5 项任务。本轮只做固定样本、固定节点的五档扰动和四种模型比较；下列数值均来自实际推理和原两阶段优化求解。

## 1. 固定设置和输入核验

| 项目 | 实际设置 |
| --- | --- |
| 代码 | 基于 `fix/cost-evaluation` 的修复代码，独立分支 `analysis/cnn-temperature-response` |
| 数据与电网 | 官方 TX-123BT 2019，case14；14 个预处理文件，每个 8760 小时 |
| 种子、删除规则 | 42；helpful；删除敏感样本 981/4906（20% 向下取整），剩余 3925 |
| 测试样本 | test index {m['reference_test_index']}，全年行 {m['reference_test_index']}；Day={m['reference_test_index']//24+1}, Hour={m['reference_test_index']%24+1}，未推断数据时区 |
| 扰动节点 | 电网节点 {s['grid_bus']}；原始 TX 零基索引 {s['tx_bus_index0']}，TX 标签 {s['tx_bus_label1']} |
| 温度位置 | `{s['tensor_slot']}`；节点内零基特征 4；140 维展开零基索引 {s['flattened_index0']} |
| σ 来源 | 原始训练池 7008 行（全年零基行 1752..8759）的温度总体标准差，ddof=0；未使用测试行计算 σ |
| σ 数值 | 编码坐标 {s['train_sigma_encoded']:.12f}，对应物理温差 {s['train_sigma_kelvin']:.9f} K（温差数值也等于 °C） |
| 原始全年标准化 | 年均值 {s['original_annual_mean_kelvin']:.9f} K，全年样本标准差 {s['original_annual_sample_std_kelvin']:.9f} K（ddof=1） |
| 参考温度 | {s['reference_temperature_kelvin']:.9f} K；模型实际 float32 编码 {s['reference_encoded_float32']:.12f} |
| TAMU | Cost 分数；平均 L1 ≤0.1，L∞≤1；与原 `eval_unchange.py` 相同 LP 和加权 CG |

输入为 `(batch, channel=1, bus=14, feature=10)`；文件按 bus 编号数值升序排列。每个节点的特征依次为：星期 sin、星期 cos、小时 sin、小时 cos、温度、短波辐射、长波辐射、纬向风速、经向风速、风速。**64 维是 CNN 学到的 tanh 隐特征，原始输入有 140 个编码值。** 完整结构及训练集统计见 [140 行输入结构表](input_structure_140.csv)，输出反标准化参数见 [14 节点缩放表](output_scaling_14.csv)。

本次从全部 365 个官方气候工作簿读取选中 TX 节点的 8760 个真实温度，逐小时验证其全年标准化值与预处理 CSV 一致；保存原温度向量及各文件 SHA256。σ 在实际训练 float32 输入上计算，Kelvin 温差与原温度训练段核对一致。五档输入只改变一个坐标，float32 舍入后的实际步长在结果表中记录。

## 2. 输入如何到达预测和 Cost

![输入输出与成本路径](figures/input_to_cost.png)

调用位置：

- [clean_data.py](../../clean_data.py)：气候取行、全年天气 z-score、特征列顺序。
- [utils/dataset.py](../../utils/dataset.py)：文件排序、`(N,1,14,10)` 排列、前 1752 行为测试、后 7008 行为训练；目标先按节点峰值映射 case14，再除 baseMVA=100，再做全年均值/标准差缩放。
- [utils/net.py: NN_CONV.forward](../../utils/net.py#L32)：卷积→ReLU→卷积→ReLU→280 维→64 维 tanh。预测头输出 14 维标准化负荷。
- [func_operation.py](../../func_operation.py)：固定 core 提取敏感特征、生成删除集合、拟合预测头、CG 遗忘模块。
- [eval_unlearn.py](../../eval_unlearn.py)、[eval_unchange.py](../../eval_unchange.py)：原始/直接/重训与 TAMU 的恢复协议。
- [utils/funcs.py: evaluate](../../utils/funcs.py)：`forecast_pu = prediction_z * target_std + target_mean`；MSE 使用 p.u.²，MAPE 使用百分比。
- [utils/optimization.py](../../utils/optimization.py)：负预测按原逻辑截为 0；第一阶段使用预测负荷得到 PG 和 cost1；第二阶段使用同一真实负荷及 PG 得到 cost2。总成本 `cost1+cost2`。

第一阶段为二次/一次发电费用及 LS1 费用；第二阶段为 LS2 和 GS 的一次/二次费用。λ=3500、κ=1250、ρ=700、η=250；发电上限、线路限制、功率平衡、非负与参考角约束均沿用原程序。七项费用和 PG/LS1/LS2/GS/角度全部保存在 [逐场景求解轨迹](cost_traces.json)。Cost 采用仓库目标函数单位，不额外声称为现实市场电价。

## 3. 三张扰动图和实际结果

### 图 1：原始 CNN 的 14 节点负荷响应

![温度与预测](figures/01_temperature_prediction.png)

预测变化以各模型自己的零扰动预测为基准。14 节点逐值数据见 [predictions_14_buses.csv](predictions_14_buses.csv)。

### 图 2：原始 CNN 的温度—Cost 路径

![温度与成本](figures/02_temperature_cost.png)

| 扰动 / σ | 总预测负荷变化 / MW | MSE / p.u.² | MAPE / % | cost1 | cost2 | 总 Cost |
| --- | --- | --- | --- | --- | --- | --- |
{original_table}

原始模型 +1σ 的 cost1 变化为 {plus_original.cost1_delta:+.6f}，cost2 变化为 {plus_original.cost2_delta:+.6f}；此次 Cost 变化主要来自第二阶段。七项费用分解显示主要为 GS 过量发电处理费用变化，LS1/LS2 的变化接近求解数值误差。此现象只对应当前固定样本。

### 图 3：四种模型的响应对比

![四种模型对比](figures/03_four_model_comparison.png)

| 模型 | 零扰动总 Cost | +1σ 总预测变化 / MW | +1σ ΔCost | −1σ ΔCost |
| --- | --- | --- | --- | --- |
{chr(10).join(model_table)}

原始模型 +1σ 时，变化最大的输出是节点 {summaries['original']['plus1_largest_response_bus']}，变化 {summaries['original']['plus1_largest_response_mw']:+.6f} MW。所有发生超过 1e−6 MW 变化的输出节点与中心差分结果见 [summary.json](summary.json)。

直接遗忘与重新拟合的五档总 Cost 最大差异为 {direct_gap:.9f}，因此曲线近乎重叠。TAMU 在本样本 +1σ 下的 ΔCost 相比直接遗忘相差 {tamu_vs_direct:+.6f}。这是一条实际局部响应曲线，是否更稳健应结合全部五档的幅度和绝对成本看，不能仅凭单个端点推出普遍结论。

**模型恢复核验：** 前三种头及 core 与上一轮已保存参数逐位相同；原始、直接遗忘和重训的全测试集 MSE/MAPE 与历史结果一致。TAMU 重建后全测试 MSE={m['full_test_validation']['tamu20']['mse']:.12f}、MAPE={m['full_test_validation']['tamu20']['mape']:.9f}%、平均 Cost={m['full_test_validation']['tamu20']['cost']:.9f}，与历史 L1=0.1 结果一致；加权 CG 与独立加权最小二乘参数最大差异 {m['tamu']['max_parameter_gap_vs_exact_weighted_fit']:.9g}。

{core_scope}

## 4. 验证、改动和复现

新增独立 [experiments/input_sensitivity.py](../../experiments/input_sensitivity.py)、[绘图与报告脚本](../../experiments/report_temperature_response.py)、[结果验证脚本](../../experiments/validate_temperature_response.py) 和相应测试。`train_nn.py`、`eval_unlearn.py`、`eval_unchange.py`、清洗/网络/优化等原有数学源码均未修改，其 SHA256 在 manifest 中记录。

最小测试先运行原始模型五档；通过后运行四种模型。零扰动预测逐位恢复；零扰动 Cost 误差 ≤1e−6；全部输出有限且维度为 14；真实负荷和模型状态保持不变。四模型主实验 20 组，共 {manifest['study_checks']['qp_solves']} 次 QP（含每模型一次独立零扰动基准）；所有求解状态 optimal，最大约束残差 {manifest['study_checks']['max_constraint_violation']:.3g}，最大目标分解误差 {manifest['study_checks']['max_objective_error']:.3g}。详细验证及自动化测试记录见 [validation.json](validation.json)、[test-results.txt](test-results.txt)。

TAMU 的 3925 变量重加权 LP 超过现有 GUROBI 许可证规模限制，自动改由 {m['tamu']['lp_solver']} 解同一个 CVXPY LP（未改变目标或约束）；两阶段小型 QP 的实际求解器在 CSV 中逐行记录。CG 返回 info=0。

使用现有 `.venv`，从仓库根目录运行：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 已有完整数据、core 权重、COST influence 和 TAMU 分数时重建
.venv/bin/python experiments/input_sensitivity.py --only-original --output-dir /tmp/temperature-minimal-new
.venv/bin/python experiments/input_sensitivity.py --output-dir /tmp/temperature-full-new
# 下载包中保存了实际输入/训练σ/四种权重，无需重新训练即可复算
.venv/bin/python experiments/input_sensitivity.py --replay-from research/cnn_temperature_response --output-dir /tmp/temperature-replay-new
.venv/bin/python experiments/report_temperature_response.py
.venv/bin/python experiments/validate_temperature_response.py --replay-dir /tmp/temperature-replay-new
.venv/bin/python -m pytest -q
```

所有输出目录须为空，避免覆盖历史结果。完整环境安装与依赖版本沿用 [成本复现报告](../../COST_REPRODUCTION_REPORT.md) 和 [requirements.txt](../../requirements.txt)。

## 5. 可汇报的结论与边界

1. 温度位于原始输入的第 5 个节点内特征，通过固定 CNN 的 64 个隐特征和 14 输出仿射头改变预测，再经两阶段调度改变 Cost；各阶段实际数据均已保存。
2. 原始、直接遗忘、重训和 TAMU 对同一输入变化的响应可以实测比较。TAMU 的权重由原测试集平均 Cost 的一阶分数优化，未对本次温度场景专门调参，也不保证每个场景的 Cost 或敏感性都降低。
3. 本轮为单节点、单样本、单种子、有限幅度的数值干预；真实负荷固定。它不代表实际发生的天气变化，不建立温度对真实负荷的因果关系，不支持跨时刻、季节、电网或 ERCOT 的总体结论。
4. 原项目先做全年天气/目标标准化再划分训练测试，存在预处理使用测试期统计的限制；本轮未重新拟合归一化以免改变模型语义，但新增扰动 σ 严格仅取训练段。helpful 删除及 TAMU 也使用测试性能信息，应视为原论文实验协议的局限。

所有三张响应图及流程图同时提供 PNG/PDF；底层数据见 [results.csv](results.csv)，冻结权重、真实输入、训练统计、源文件校验及命令均在本目录。[从 GitHub 下载完整 ZIP](https://github.com/syh-Mason/task_aware_machine_unlearning/blob/analysis/cnn-temperature-response/exports/cnn_temperature_response.zip?raw=true)。
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    brief=f'''# 导师汇报：CNN 温度输入如何影响电网 Cost

**本轮完成：** 核对输入输出、单特征扰动、两阶段 Cost 跟踪、四模型对照和可复现实验。

固定官方 TX-123BT/case14、种子 42、helpful 删除 20%、TAMU L1=0.1，取测试样本 {m['reference_test_index']}、电网节点 {s['grid_bus']}。温度加上 −1、−0.5、0、+0.5、+1 个训练集标准差；σ={s['train_sigma_kelvin']:.6f} K，仅由原训练段计算。真实负荷和系统约束固定。

**1. 温度在什么位置？** 输入是 14 节点×10 特征，温度为每节点第 5 项；本节点位置 `{s['tensor_slot']}`。64 维是隐藏表示。原全年归一化未改，训练 σ 在现有编码上计算并与官方原温度核对。

**2. 哪些预测发生变化？** +1σ 时，原始模型全部 14 个输出节点变化超过 1e−6 MW；总预测负荷变化 {summaries['original']['plus1_total_prediction_delta_mw']:+.6f} MW。变化最大为节点 {summaries['original']['plus1_largest_response_bus']}：{summaries['original']['plus1_largest_response_mw']:+.6f} MW。见图 1 和逐节点 CSV。

**3. 如何传递到 Cost？** 温度→冻结卷积核心→64 隐特征→14 维预测→反标准化→阶段一发电调度→阶段二按固定真实负荷再调度→cost1+cost2。原始模型 +1σ 的 cost1 变化 {plus_original.cost1_delta:+.6f}、cost2 变化 {plus_original.cost2_delta:+.6f}，主要是第二阶段过量发电处理费用变化。七项费用与调度变量均可追踪。见流程图和图 2。

**4. 遗忘前后有什么不同？**

| 模型 | 零扰动 Cost | +1σ 总预测变化/MW | +1σ ΔCost | −1σ ΔCost |
| --- | --- | --- | --- | --- |
{chr(10).join(model_table)}

直接遗忘与重训响应几乎重合。TAMU 的 +1σ 响应幅度较小，但 −1σ 的 Cost 响应幅度更大；不能称为所有方向都更稳健。五档曲线也不是温度与预测的简单单调关系。

**实验边界：** 沿用原项目冻结 core、仅重训/遗忘仿射头的协议。此为单样本、单节点、单种子的数值干预，不代表实际天气事件或因果结论。原项目全年标准化及测试指标选样的局限已在完整报告说明。

**验证与交付：** 20 组实际模型场景、280 个逐节点预测；零扰动恢复预测和 Cost；全部求解 optimal；原成本目标及约束未改。提供三张响应图、一张流程图（PNG/PDF）、独立运行脚本、冻结权重/真实输入、CSV 和自动化验证。完整测试通过数见 [test-results.txt](test-results.txt)。

- [图 1：温度与预测](figures/01_temperature_prediction.png)
- [图 2：温度与 Cost](figures/02_temperature_cost.png)
- [图 3：四模型响应](figures/03_four_model_comparison.png)
- [输入输出流程图](figures/input_to_cost.png)
- [完整报告与复现命令](README.md)
'''
    (out/'导师汇报.md').write_text(brief,encoding='utf-8')


if __name__=='__main__':
    main()

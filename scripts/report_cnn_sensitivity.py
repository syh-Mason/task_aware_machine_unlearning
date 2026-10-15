"""Render measured CNN sensitivity results, source diagrams and a Chinese report."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_cnn_sensitivity import write_csv, write_json

NAMES = {'original':'Original CNN head', 'unlearned20':'After 20% unlearning'}
COLORS = {'original':'#2459b3', 'unlearned20':'#db5937'}
SHORT = ['Weekday sin','Weekday cos','Hour sin','Hour cos','Temperature',
         'Shortwave','Longwave','Zonal wind','Meridional wind','Wind speed']


def saved(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def labelled(row, space):
    return f'z{int(row.input_index1):02d}' if space=='latent64' else f'节点{int(row.grid_bus)} / {row.feature}'


def rank_comparison(a, b, score, space):
    a, b = a.sort_values('input_index1'), b.sort_values('input_index1')
    x = a[score].abs() if score=='cost_delta' else a[score]
    y = b[score].abs() if score=='cost_delta' else b[score]
    top_a, top_b = set(a.loc[x.nlargest(10).index].input_index1), set(b.loc[y.nlargest(10).index].input_index1)
    def top(rows, values):
        return [dict(input_index1=int(rows.loc[i].input_index1), label=labelled(rows.loc[i],space), value=float(values.loc[i]))
                for i in values.nlargest(5).index]
    rho = float(spearmanr(x.to_numpy(), y.to_numpy()).statistic)
    return dict(spearman=rho if np.isfinite(rho) else None, top10_overlap=len(top_a&top_b),
                original_top5=top(a,x), unlearned20_top5=top(b,y))


def sensitivity_charts(data, fixed, aggregated, figures):
    reference_index=int(fixed.test_index.iloc[0])
    # This study uses +10%; read the actual measured baseline/step for labels.
    for space, size in [('latent64',64),('raw140',140)]:
        if space=='latent64':
            for score, ylabel, suffix in [('prediction_mae_mw','Mean absolute forecast change per bus (MW)','prediction'),
                                           ('cost_delta','Change in total cost (repository units)','cost')]:
                fig, ax = plt.subplots(figsize=(13,4.4))
                for model in NAMES:
                    rows=fixed[(fixed.space==space)&(fixed.model==model)].sort_values('input_index1')
                    ax.plot(rows.input_index1,rows[score],'.-',color=COLORS[model],label=NAMES[model],linewidth=1.1,markersize=4)
                ax.axhline(0,color='0.4',linewidth=.7)
                ax.set(xlabel='Learned tanh coordinate (z01 ... z64)',ylabel=ylabel,
                       title=f'Fixed recorded test sample {reference_index}: one hidden coordinate multiplied by 1.1')
                ax.set_xticks([1,*range(4,65,4)]);ax.legend();ax.grid(alpha=.25)
                saved(fig,figures,f'latent64_{suffix}_sensitivity')
        else:
            for score, label, suffix in [('prediction_mae_mw','Mean absolute forecast change per bus (MW)','prediction'),
                                         ('cost_delta','Signed change in total cost','cost')]:
                original=fixed[(fixed.space==space)&(fixed.model=='original')].sort_values('input_index1')[score].to_numpy().reshape(14,10)
                unlearned=fixed[(fixed.space==space)&(fixed.model=='unlearned20')].sort_values('input_index1')[score].to_numpy().reshape(14,10)
                fig,axes=plt.subplots(1,3,figsize=(15,6.7),layout='constrained')
                scale=max(np.abs(original).max(),np.abs(unlearned).max())
                difference=unlearned-original
                diffscale=max(np.abs(difference).max(),1e-15)
                for index,(ax,values,title) in enumerate(zip(axes,[original,unlearned,difference],
                        ['Original CNN head','After 20% unlearning','After minus original'])):
                    signed=score=='cost_delta' or index==2
                    bound=diffscale if index==2 else scale
                    im=ax.imshow(values,cmap='RdBu_r' if signed else 'viridis',vmin=-bound if signed else 0,vmax=bound,aspect='auto')
                    ax.set_title(title);ax.set_xticks(range(10),SHORT,rotation=65,ha='right')
                    ax.set_yticks(range(14),[f'Grid {i:02d}' for i in range(1,15)])
                    fig.colorbar(im,ax=ax,shrink=.7,label=label)
                fig.suptitle(f'Fixed sample {reference_index}: each of the 140 encoded input coordinates perturbed separately (+10%)')
                saved(fig,figures,f'raw140_{suffix}_heatmap')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,space in enumerate(('latent64','raw140')):
        for row,(score,label) in enumerate([('prediction_mae_mw','Prediction-response rank'),('cost_delta','Absolute cost-response rank')]):
            a=fixed[(fixed.space==space)&(fixed.model=='original')].sort_values('input_index1')
            b=fixed[(fixed.space==space)&(fixed.model=='unlearned20')].sort_values('input_index1')
            x=a[score].abs().rank(ascending=False,method='average').to_numpy()
            y=b[score].abs().rank(ascending=False,method='average').to_numpy()
            ax=axes[row,col];ax.scatter(x,y,s=18,alpha=.7)
            ax.plot([1,len(x)],[1,len(x)],'--',color='0.5')
            ax.set(xlabel='Original rank (1 = largest)',ylabel='After 20% rank',title=f'{space}: {label}')
            ax.grid(alpha=.2)
    saved(fig,figures,'before_after_rank_comparison')
    fig,axes=plt.subplots(1,2,figsize=(14,4.6),layout='constrained')
    for score,ax,label in [('prediction_mean_mw',axes[0],'Mean absolute forecast response (MW)'),
                           ('cost_abs_mean',axes[1],'Mean absolute cost response')]:
        for model in NAMES:
            rows=aggregated[(aggregated.space=='latent64')&(aggregated.model==model)].sort_values('input_index1')
            ax.plot(rows.input_index1,rows[score],'.-',label=NAMES[model],color=COLORS[model],linewidth=1)
        ax.set(xlabel='Hidden coordinate',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Descriptive means across {data.context.nunique()} fixed recorded test times')
    saved(fig,figures,'cohort24_hidden_sensitivity')
    raw=data[data.space=='raw140'].copy();raw['feature_index0']=(raw.input_index1-1)%10
    grouped=raw.groupby(['model','feature_index0']).agg(prediction=('prediction_mae_mw','mean'),cost=('abs_cost_delta','mean')).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(14,5.7),layout='constrained')
    for score,ax,label in [('prediction',axes[0],'Mean absolute forecast response (MW)'),('cost',axes[1],'Mean absolute cost response')]:
        for offset,model in enumerate(NAMES):
            rows=grouped[grouped.model==model].sort_values('feature_index0')
            ax.bar(np.arange(10)+(offset-.5)*.36,rows[score],width=.36,color=COLORS[model],label=NAMES[model])
        ax.set_xticks(range(10),SHORT,rotation=50,ha='right');ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Feature-type means: average of separate single-coordinate interventions, across 14 buses and recorded contexts')
    saved(fig,figures,'physical_feature_type_comparison')
    return grouped


def source_anchors():
    targets={'clean_data.py':['main'], 'utils/dataset.py':['MyDataset','case_modifier'],
             'utils/net.py':['NN_CONV.forward'], 'func_operation.py':['return_dataset_for_nn_affine','return_unlearn_datasets'],
             'utils/funcs.py':['evaluate'], 'utils/optimization.py':['Operator.stage_one','Operator.stage_two','Operator.solve_one','Operator.solve_two'],
             'eval_unlearn.py':['main']}
    links={}
    for filename,names in targets.items():
        tree=ast.parse((ROOT/filename).read_text())
        for name in names:
            nodes=tree.body
            for part in name.split('.'):
                node=next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part)
                nodes=node.body
            links[f'{filename}:{name}']=f'[{filename}:{node.lineno}](../../{filename}#L{node.lineno})'
    return links


def diagrams(output):
    common='node [shape=box, style="rounded,filled", fillcolor="#edf4fb", fontname="Noto Sans CJK JP", fontsize=12]; edge [fontname="Noto Sans CJK JP",fontsize=10]; graph [fontname="Noto Sans CJK JP",bgcolor="white",pad="0.25",nodesep="0.25",ranksep="0.4"];'
    cnn='''digraph CNN {
    rankdir=TB;
    raw [label="TX-123BT 2019\n每日负荷 + 气候工作簿"];
    csv [label="clean_data.py\n14个CSV：各8760行 × (10特征 + Load)"];
    data [label="MyDataset / return_dataset\n单样本 x: (1, 1, 14, 10)，目标 y: (1, 14)"];
    conv1 [label="NN_CONV.forward\nconv1 + ReLU → (1, 8, 14, 10)"];
    conv2 [label="conv2 + ReLU → (1, 8, 7, 5)\nFlatten → (1, 280)"];
    latent [label="lin1 + tanh → z: (1, 64)\n64维是学习到的隐藏表示"];
    original [label="敏感集4906条解析拟合的仿射头\nW_original: 14×64，b: 14"];
    unlearn [label="20%遗忘：删除981，保留3925\nCG更新最后仿射头；卷积提取器固定"];
    native [label="core.pth自带lin2 → (1,14)\n用于核心网络训练\n遗忘评估取 feature[0] 后另拟合头",fillcolor="#eeeeee",color="gray"];
    forecast [label="预测标准化负荷 (1,14)\ny_hat = W z + b"];
    raw -> csv -> data -> conv1 -> conv2 -> latent;
    latent -> original -> forecast; latent -> unlearn -> forecast;
    latent -> native [style=dashed,color=gray];
    }
    '''
    cost='''digraph COST {
    rankdir=TB;
    pred [label="仿射头预测：y_hat (14维标准化值)"];
    scale [label="evaluate / Operator.solve_one\nd_hat = y_hat * target_std + target_mean (p.u.)\nd_hat_clip = max(d_hat, 0)"];
    dispatch [label="阶段一：以预测负荷调度\nmin PGᵀdiag(q)PG + cᵀPG + λΣLS1\n发电限额、双向线路限额、节点功率平衡\nLS1≥0、参考节点角固定"];
    truth [label="固定同一观测真实负荷\nd_true = target_std*y_true + target_mean\n输入扰动与机器遗忘均不修改它",fillcolor="#e8f4e5"];
    redispatch [label="阶段二：给定PG与真实负荷再调度\nmin λΣLS2 + κΣLS2² + ρΣGS + ηΣGS²\n节点平衡、双向线路限额、LS2/GS≥0、参考角固定"];
    total [label="单样本 cost = cost1 + cost2\n响应 Δcost = cost_perturbed - cost_baseline\n每个模型与其自己的基线比较"];
    pred -> scale -> dispatch -> redispatch -> total;
    truth -> redispatch;
    }
    '''
    for name,body in [('cnn_call_path',cnn),('forecast_to_cost',cost)]:
        text=body.replace('{','{\n'+common,1)
        text='\n'.join(line.rstrip() for line in text.splitlines()).rstrip()+'\n'
        path=output/f'{name}.dot';path.write_text(text,encoding='utf-8')
        for extension in ('png','pdf','svg'):
            subprocess.run(['dot',f'-T{extension}',str(path),'-o',str(output/f'{name}.{extension}')],check=True)


def report(directory, manifest, fixed, aggregated, summary, grouped):
    links=source_anchors();counts=manifest['model_validation']['counts'];v=manifest['study_validation']
    base=fixed[fixed.space=='baseline'].set_index('model')
    latent_peak=summary['fixed']['latent64']['prediction_mae_mw']
    raw_original=fixed[(fixed.space=='raw140')&(fixed.model=='original')]
    raw_peak=raw_original.loc[raw_original.prediction_mae_mw.idxmax()]
    raw_after=fixed[(fixed.space=='raw140')&(fixed.model=='unlearned20')&(fixed.input_index1==raw_peak.input_index1)].iloc[0]
    cost_example=[];cost_table=[]
    components=['dispatch_generation_quadratic','dispatch_generation_linear','dispatch_shedding',
                'redispatch_shedding_linear','redispatch_shedding_quadratic','redispatch_storage_linear','redispatch_storage_quadratic']
    numeric=['prediction_total_change_mw','cost1','cost2','cost_total','ls1_total','ls2_total','storage_total',*components]
    for model in NAMES:
        baseline=fixed[(fixed.model==model)&(fixed.space=='baseline')].iloc[0]
        perturbed=fixed[(fixed.model==model)&(fixed.space=='raw140')&(fixed.input_index1==raw_peak.input_index1)].iloc[0]
        for label,values in [('baseline',baseline),('perturbed',perturbed),('difference',perturbed[numeric]-baseline[numeric])]:
            cost_example.append(dict(model=model,case=label,input_index1=int(raw_peak.input_index1),
                                **{column:float(values[column]) for column in numeric}))
        delta=perturbed[numeric]-baseline[numeric]
        cost_table.append(f"| {model} | {delta.prediction_total_change_mw:+.6f} | {delta.cost1:+.6f} | {delta.cost2:+.6f} | {delta.cost_total:+.6f} | {delta.storage_total*v['baseMVA']:+.6f} |")
    write_csv(directory/'cost_path_largest_raw_example.csv',cost_example)
    rows=[]
    for space in ('latent64','raw140'):
        for name,label in [('prediction_mae_mw','预测响应'),('cost_delta','Cost响应')]:
            s=summary['fixed'][space][name]
            rho=f"{s['spearman']:.6f}" if s['spearman'] is not None else '未定义（响应为常量）'
            rows.append(f"| {space} {label} | {rho} | {s['top10_overlap']}/10 | {s['original_top5'][0]['label']} | {s['unlearned20_top5'][0]['label']} |")
    comparison='\n'.join(rows)
    top_rows=[]
    for space in ('latent64','raw140'):
        for model in NAMES:
            sub=fixed[(fixed.space==space)&(fixed.model==model)]
            point=sub.loc[sub.prediction_mae_mw.idxmax()]
            top_rows.append(f"| {space} | {model} | {labelled(point,space)} | {point.prediction_mae_mw:.9f} | {point.cost_delta:+.9f} |")
    first=fixed[(fixed.space=='latent64')&(fixed.input_index1==1)].set_index('model')
    first_raw=fixed[(fixed.space=='raw140')&(fixed.input_index1==1)].set_index('model')
    validation=manifest['model_validation']['full_test_validation']
    proof_path=directory/'validation.json'
    proof=json.loads(proof_path.read_text()) if proof_path.is_file() else {}
    proof_text=(f"已通过{proof['tests_passed']}项自动化测试，0失败、0跳过；" if proof.get('tests_passed') else '')
    if proof.get('all_replay_forecasts_bit_identical') and proof.get('all_intervention_csv_replay_byte_identical'):
        proof_text+=f"快照完整重放的{proof.get('replay_forecast_values',137760)}个预测值逐位一致，全部干预CSV逐字节一致。"
    if proof:
        proof_text+='完整验证记录见 [validation.json](validation.json)，测试输出见 [test-results.txt](test-results.txt)。'
    document=f'''# CNN 输入—预测—Cost 与 20% 遗忘敏感性研究

五项任务已经完成；本页和关联 CSV/PNG/PDF 均来自现有真实检查点与官方 TX-123BT 数据。TX-123BT 是公开的合成系统数据，本次使用其已存在的测试样本。分析使用固定卷积提取器，重新按原算法构建敏感集仿射头及 20% CG 遗忘头，没有重新训练 CNN 核心网络，也没有更改 cost、目标函数或数学约束。

- [x] 输入含义：140维原始特征表与64维隐藏表示表。
- [x] CNN路径：源码定位、实测张量形状和调用流程图。
- [x] 单变量扰动：每次一个坐标乘1.1，记录完整14维预测与敏感性图。
- [x] Cost路径：逐次求解原两阶段问题，记录费用分量和约束残差。
- [x] 遗忘前后：固定提取器和真实负荷，对比20%遗忘后的响应幅度及排名。

固定样本的实测结果：原始与遗忘模型最敏感的隐藏坐标分别为 {latent_peak['original_top5'][0]['label']} / {latent_peak['unlearned20_top5'][0]['label']}，对应预测响应 {latent_peak['original_top5'][0]['value']:.6f} / {latent_peak['unlearned20_top5'][0]['value']:.6f} MW；64维预测响应的Top10重合{latent_peak['top10_overlap']}项。原始编码输入中，原模型响应最大的坐标为{labelled(raw_peak,'raw140')}，同一坐标前后干预Δcost为{raw_peak.cost_delta:+.6f} / {raw_after.cost_delta:+.6f}。排名与幅度应结合完整表评价，单个最大响应坐标相同不代表整体敏感性不变。

## 一、确认输入维度与实际意义

**原始 CNN 输入不是64维，而是14个节点 × 10类特征 = 140维。64维是 `lin1 → tanh` 输出的学习表示。** 各隐藏维混合了原始特征，没有可直接命名为“温度”“风速”的逐维物理对应。其编号只能写为 z01…z64。

| 类别 | 10类原始特征（每个节点均有） | CSV实际存储值 |
| --- | --- | --- |
| 时间 | Weekday_sin、Weekday_cos、Hour_sin、Hour_cos | sin/cos编码，无量纲；weekday从0开始，hour标签1…24 |
| 气象 | Temperature (k)、Shortwave Radiation (w/m2)、Longwave Radiation (w/m2)、Zonal Wind Speed (m/s)、Meridional Wind Speed (m/s)、Wind Speed (m/s) | 每个原始TX节点按全年均值和样本标准差(ddof=1)进行z-score，单位已经消除 |

气象原单位分别是 K、W/m²、W/m²、m/s、m/s、m/s。CSV保留原列名，但数值已标准化。`Load` 是目标列，不进入10个输入特征。实际模型没有64个滞后负荷输入。隐藏特征经过 tanh，原始范围为[-1,1]；最后仿射层使用64个特征，另有偏置，共14×(64+1)=910个可遗忘参数。

- [140维输入说明表](raw_input_features_140.csv)：每个坐标的节点号、字段、原单位、存储编码、原始TX索引及当前样本取值。
- [64维隐藏表示说明与实测敏感性表](latent_features_64.csv)：逐维激活、干预步长、线性头单位增益及前后排名；不赋予未经证实的物理语义。
- 原始TX的CSV文件 `bus_k.csv` 使用0基列索引，源气候 Bus 标签为k+1。加载器按文件名数字排序后映射至14节点模型，完整对应已写入输入表。
- 源码：{links['clean_data.py:main']}、{links['utils/dataset.py:MyDataset']}、{links['utils/dataset.py:case_modifier']}。

## 二、CNN 前向与输出调用链

![CNN调用路径](figures/cnn_call_path.png)

[矢量PDF](figures/cnn_call_path.pdf) · [SVG](figures/cnn_call_path.svg) · [真实前向张量形状](forward_trace.json)。卷积参数为1→8通道、3×3/stride1；第二层8→8通道、4×4/stride2；Flatten为280维；lin1为280→64；tanh输出64维。

`train_nn.py` 的核心网络预测取 `net(x)[1]`，对应 checkpoint 内的 lin2。遗忘实验的 {links['func_operation.py:return_dataset_for_nn_affine']} 取 `net(x)[0]`，再在敏感集上解析拟合最后仿射头。因此本报告的“原始CNN”指现有遗忘流程中的**固定CNN提取器 + 未遗忘的敏感集仿射头**。20%遗忘更新这个头，提取器不变；没有把 core.pth 自带 lin2 与另一个头混作前后对照。{links['utils/net.py:NN_CONV.forward']}、{links['eval_unlearn.py:main']}。

数据为8760小时；训练{counts['train']}，测试{counts['test']}。测试集实际是全年最前1752小时，训练集是之后7008小时；核心集{counts['core']}，敏感集{counts['sensitive']}，20%删除{counts['deleted']}、保留{counts['remain']}。候选样本按原默认 cost influence 从最有帮助的31%中按seed42选取，详见 {links['func_operation.py:return_unlearn_datasets']}。两个模型重建后的全测试集MSE/MAPE与此前保存结果相符：原模型MSE={validation['original']['mse']:.12f}、MAPE={validation['original']['mape']:.9f}%；遗忘模型MSE={validation['unlearned20']['mse']:.12f}、MAPE={validation['unlearned20']['mape']:.9f}%。

## 三、单变量 +10% 扰动

固定展示样本为 test index 0（原始年度行0：Day1、Hour1标签，没有额外推断时区）。逐次令 x_j'=1.1x_j，其余输入坐标、模型参数及同一真实负荷全部固定。分别执行64维隐藏坐标干预和140维原始编码坐标干预；敏感性主要度量为14个节点预测变化绝对值的均值，换算MW使用case14的baseMVA={v['baseMVA']:g}。

**乘1.1对负数表示绝对值增加10%，对零不产生变化。天气z-score乘1.1表示离全年均值的距离增加10%，不等于温度/风速物理值增加10%。独立修改时间sin/cos会破坏共同时间及单位圆关系；隐藏值也可能越过tanh范围。** 这些是冻结模型的数值敏感性探针，不作为可实现天气或时间反事实。本次有{v['coordinate_outside_encoded_bound_count']}条模型干预记录超出其编码单位界，已逐条标记，未裁剪或隐瞒。

![64维预测敏感性](figures/latent64_prediction_sensitivity.png)

![140维原始输入预测敏感性](figures/raw140_prediction_heatmap.png)

第一个隐藏坐标 z01 的实际值与步长分别为 {first.loc['original','input_before']:.9f}、{first.loc['original','actual_step']:+.9f}。其原模型/遗忘模型平均预测响应分别为 {first.loc['original','prediction_mae_mw']:.9f} / {first.loc['unlearned20','prediction_mae_mw']:.9f} MW，对应Δcost为 {first.loc['original','cost_delta']:+.9f} / {first.loc['unlearned20','cost_delta']:+.9f}。原始第1坐标是节点1的 Weekday_sin，响应分别为 {first_raw.loc['original','prediction_mae_mw']:.9f} / {first_raw.loc['unlearned20','prediction_mae_mw']:.9f} MW。完整14维预测向量在 [所有预测NPZ](all_predictions.npz) 和 [首坐标示例表](first_coordinate_examples.csv) 中。

![首坐标逐节点预测变化](figures/first_coordinate_prediction_changes.png)

## 四、预测到 Cost 的完整路径

![预测到cost](figures/forecast_to_cost.png)

[矢量PDF](figures/forecast_to_cost.pdf) · [逐项费用、状态、约束残差与决策变量](fixed_sample_cost_trace.json)。预测先按目标的全年mean/std还原为p.u.，负预测按原代码截断为0；原始负荷先按每节点全年最大值映射到case14标准负荷，再除baseMVA。标准化顺序、截断和两阶段均保持原实现。目标标准差使用NumPy总体标准差(ddof=0)，区别于天气输入的Pandas样本标准差(ddof=1)。

第一阶段最小化 `PGᵀdiag(q)PG + cᵀPG + λΣLS1`；保留发电上下限、线路双向限额、`Bbusθ=CgPG−Cl(d_hat−LS1)`、LS1≥0和参考角。

第二阶段固定第一阶段PG，最小化 `λΣLS2 + κΣLS2² + ρΣGS + ηΣGS²`；保留 `Bbusθ=Cg(PG−GS)−Cl(d_true−LS2)`、双向线路限额、LS2≥0、GS≥0和参考角。case14 λ=3500、κ=1250、ρ=700、η=250；不添加原项目没有的GS上界或LS上界。

单样本cost等于cost1+cost2，本报告沿用仓库目标函数数值。每次干预使用固定真实负荷，且每个模型的Δcost都相对于该模型自身的基线，不把遗忘本身的基线差混入扰动响应。源码：{links['utils/funcs.py:evaluate']}、{links['utils/optimization.py:Operator.solve_one']}、{links['utils/optimization.py:Operator.solve_two']}。

| 固定样本 | 阶段一 | 阶段二 | 总cost |
| --- | ---: | ---: | ---: |
| 原模型 | {base.loc['original','cost1']:.9f} | {base.loc['original','cost2']:.9f} | {base.loc['original','cost_total']:.9f} |
| 遗忘20% | {base.loc['unlearned20','cost1']:.9f} | {base.loc['unlearned20','cost2']:.9f} | {base.loc['unlearned20','cost_total']:.9f} |

从已完成的原始输入干预中，选取本样本预测响应最大的坐标（{labelled(raw_peak,'raw140')}）展示费用路径。其编码值从{raw_peak.input_before:.9f}变为{raw_peak.input_after:.9f}；这是编码数值乘1.1，符号方向已保留。以下各列均为同一次干预相对本模型基线的变化，完整七项费用见 [费用示例CSV](cost_path_largest_raw_example.csv)。

| 模型 | 预测总负荷ΔMW | Δcost1 | Δcost2 | 净Δcost | 储存/过量发电处理ΔMW |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(cost_table)}

该固定示例中，温度编码坐标扰动使预测总负荷上升，阶段一费用与阶段二的过量发电处理费用均增加；真实负荷保持相同。这个例子的主要cost增加来自再调度项，不能只根据预测误差大小或阶段一发电成本推断总cost。

![64维cost响应](figures/latent64_cost_sensitivity.png)

![140维cost响应](figures/raw140_cost_heatmap.png)

## 五、遗忘前后敏感性变化

| 空间/指标 | 前后Spearman排名相关 | Top10重合 | 原模型最大响应坐标 | 遗忘后最大响应坐标 |
| --- | ---: | ---: | --- | --- |
{comparison}

Spearman越接近1表示排序越相似，不能据此断言响应幅度相同。预测响应与cost响应分别排名，cost按绝对变化排名，同时保留正负方向。

| 空间 | 模型 | 预测响应最大的坐标 | 平均绝对预测变化MW | 同次Δcost |
| --- | --- | --- | ---: | ---: |
{chr(10).join(top_rows)}

![遗忘前后排名](figures/before_after_rank_comparison.png)

在同一个隐藏坐标值及步长下，`Δprediction = diag(std) W Δz`。本次逐项验证误差最大为 {v['exact_latent_formula_max_error']:.3g} p.u.；原始输入的特征提取变化Δz对两个头相同，前后预测响应之差等于 `diag(std)(W_after−W_before)Δz`。因此本实验中敏感性变化可定位到最后仿射头变化；cost的后续变化还受预测基线和约束作用区域影响。该结论不涉及已冻结卷积层发生参数变化。

24个时点按当前测试集均匀选取，索引为 {manifest['model_validation']['context_test_indices']}。它们只覆盖当前测试集的最前73天，不代表全年或总体。分别计算所有坐标干预后取描述性均值，没有按最大效应挑选样本或坐标。

![24时点隐藏敏感性](figures/cohort24_hidden_sensitivity.png)

![原始特征类型对比](figures/physical_feature_type_comparison.png)

特征类型柱状图是对14个节点上的**独立单坐标干预**再跨时点取均值，并非同时改变14个输入。可用 [排名统计JSON](summary.json)、[固定样本带排名CSV](fixed_sample_ranked.csv)、[跨时点均值CSV](context_aggregated.csv) 核验。单样本和跨时点结果不可替换为因果物理重要性或全数据集统计结论。

## 数据、验证及复现

{proof_text}

- [完整干预CSV](all_interventions.csv)：{v['rows']}行，含48条基线和{v['interventions']}条干预；两阶段费用分量、求解状态/残差、预测变化和固定真实负荷路径均有记录。
- [便携冻结模型](frozen_models.npz)、[真实输入及目标快照](reference_inputs.npz)：支持不读取原始数据、不重新训练CNN的重放。核心checkpoint、影响分数、14个CSV及源文件校验值见 [manifest](manifest.json)。
- 实际共执行{v['qp_solves']}次QP，全部{', '.join(v['statuses'])}，求解器{', '.join(v['solvers'])}，最大原约束残差{v['max_constraint_violation']:.3g}，费用分解误差{v['max_objective_decomposition_error']:.3g}。
- 原模型与遗忘模型及提取器的state哈希在研究前后完全一致，所有干预保持真实负荷不变。CG收敛info=0，遗忘与重训练头最大参数差为 {manifest['model_validation']['max_parameter_gap_vs_retraining']:.9f}，保留原容差1e-5。

从仓库根目录执行：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cloud-setup/task-aware/matplotlib
.venv/bin/python scripts/analyze_cnn_sensitivity.py
.venv/bin/python scripts/report_cnn_sensitivity.py
.venv/bin/python -m pytest tests -q

# 使用导出真实快照完整重放；输出目录必须为空
.venv/bin/python scripts/analyze_cnn_sensitivity.py --replay-from research/cnn_sensitivity --output-dir /tmp/tamu-cnn-sensitivity-replay
.venv/bin/python scripts/validate_cnn_sensitivity.py --with-original-data --replay-dir /tmp/tamu-cnn-sensitivity-replay
```

局限：这是编码输入和隐藏表示的有限扰动研究，部分探针不在实际数据流形内；不等同于物理温度增加10%的研究。当前预处理在划分前使用全年统计量和最大值，保留了原方案，但限制其作为严格盲测评估的解释。尚未接入ERCOT，也没有作跨数据集、跨季节或全网络重新训练后敏感性稳定性的结论。既有cost定义和数学约束均保持不变。
'''
    (directory/'README.md').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'research/cnn_sensitivity')
    args=parser.parse_args();directory=args.results_dir.resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    data=pd.read_csv(directory/'all_interventions.csv')
    if len(data)!=manifest['study_validation']['rows'] or not np.isfinite(data[['cost_total','prediction_mae_mw','cost_delta']]).all().all():
        raise ValueError('Incomplete or nonfinite measured results')
    data['abs_cost_delta']=data.cost_delta.abs()
    fixed=data[data.context==0].copy()
    for score in ('prediction_mae_mw','abs_cost_delta'):
        fixed[f'{score}_rank']=fixed.groupby(['space','model'])[score].rank(ascending=False,method='average')
    fixed.to_csv(directory/'fixed_sample_ranked.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    interventions=data[data.space!='baseline']
    aggregated=interventions.groupby(['space','model','input_index1','feature'],dropna=False).agg(
        prediction_mean_mw=('prediction_mae_mw','mean'), prediction_std_mw=('prediction_mae_mw','std'),
        cost_abs_mean=('abs_cost_delta','mean'),cost_signed_mean=('cost_delta','mean'),cost_abs_max=('abs_cost_delta','max')).reset_index()
    aggregated.to_csv(directory/'context_aggregated.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    summary=dict(fixed={},cohort_mean={})
    for space in ('latent64','raw140'):
        summary['fixed'][space]={}
        a=fixed[(fixed.space==space)&(fixed.model=='original')];b=fixed[(fixed.space==space)&(fixed.model=='unlearned20')]
        for score in ('prediction_mae_mw','cost_delta'):
            summary['fixed'][space][score]=rank_comparison(a,b,score,space)
        # Convert aggregate column names to the same response vocabulary.
        a=aggregated[(aggregated.space==space)&(aggregated.model=='original')].rename(columns={'prediction_mean_mw':'prediction_mae_mw','cost_abs_mean':'cost_delta'})
        b=aggregated[(aggregated.space==space)&(aggregated.model=='unlearned20')].rename(columns={'prediction_mean_mw':'prediction_mae_mw','cost_abs_mean':'cost_delta'})
        if space=='raw140':
            a=a.assign(grid_bus=(a.input_index1-1)//10+1);b=b.assign(grid_bus=(b.input_index1-1)//10+1)
        summary['cohort_mean'][space]={score:rank_comparison(a,b,score,space) for score in ('prediction_mae_mw','cost_delta')}
    write_json(directory/'summary.json',summary)
    sample=json.loads((directory/'fixed_sample.json').read_text());models=np.load(directory/'frozen_models.npz',allow_pickle=False)
    latent=fixed[fixed.space=='latent64'].sort_values(['model','input_index1'])
    rows=[]
    for index in range(64):
        row=dict(coordinate1=index+1,name=f'z{index+1:02d}',meaning='learned mixed tanh feature; no fixed physical name',
                 layer='NN_CONV.lin1 -> tanh; input to sensitive affine head',normal_range='[-1,1]',reference_value=sample['hidden64'][index])
        intervention=latent[(latent.model=='original')&(latent.input_index1==index+1)].iloc[0]
        row.update(actual_step=float(intervention.actual_step),value_after=float(intervention.input_after),
                   outside_tanh_range=bool(intervention.encoded_value_outside_unit_bound))
        for model in NAMES:
            measured=latent[(latent.model==model)&(latent.input_index1==index+1)].iloc[0]
            row.update({f'{model}_prediction_change_mw':float(measured.prediction_mae_mw),f'{model}_cost_delta':float(measured.cost_delta),
                        f'{model}_prediction_rank':float(measured.prediction_mae_mw_rank),
                        f'{model}_mean_abs_gain_mw_per_unit_z':float(np.mean(np.abs(models[f'{model}.weight'][:,index]*sample['std_pu']))*sample['baseMVA'])})
        rows.append(row)
    write_csv(directory/'latent_features_64.csv',rows)
    forecasts=np.load(directory/'all_predictions.npz',allow_pickle=False)['forecasts_pu']
    example=[]
    for model_index,model in enumerate(NAMES):
        for space,case in [('latent64',1),('raw140',65)]:
            for bus in range(14):
                example.append(dict(model=model,space=space,input_index1=1,grid_bus=bus+1,
                               actual_load_mw=sample['target_mw'][bus],baseline_prediction_mw=float(forecasts[model_index,0,0,bus]*sample['baseMVA']),
                               perturbed_prediction_mw=float(forecasts[model_index,0,case,bus]*sample['baseMVA']),
                               delta_prediction_mw=float((forecasts[model_index,0,case,bus]-forecasts[model_index,0,0,bus])*sample['baseMVA'])))
    write_csv(directory/'first_coordinate_examples.csv',example)
    figures=directory/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    examples=pd.DataFrame(example)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for ax,space,label in zip(axes,('latent64','raw140'),('Hidden z01 multiplied by 1.1','Raw coordinate 1: Grid01 / Weekday sin')):
        for offset,model in enumerate(NAMES):
            rows=examples[(examples.model==model)&(examples.space==space)].sort_values('grid_bus')
            ax.bar(np.arange(1,15)+(offset-.5)*.35,rows.delta_prediction_mw,width=.35,label=NAMES[model],color=COLORS[model])
        ax.axhline(0,color='0.3',linewidth=.7);ax.set(xlabel='Forecast grid bus',ylabel='Signed forecast change (MW)',title=label)
        ax.set_xticks(range(1,15));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    saved(fig,figures,'first_coordinate_prediction_changes')
    grouped=sensitivity_charts(data,fixed,aggregated,figures)
    grouped.to_csv(directory/'physical_feature_type_means.csv',index=False,lineterminator='\n')
    diagrams(figures)
    report(directory,manifest,fixed,aggregated,summary,grouped)
    print(json.dumps(summary['fixed'],ensure_ascii=False,indent=2))
    print(f'Figures, source diagrams and measured report saved to {directory}')


if __name__=='__main__':
    main()

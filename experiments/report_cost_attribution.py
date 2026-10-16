"""Report actual output-node counterfactuals and stratified CNN cost responses."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.input_sensitivity import NAMES, LEVELS
from scripts.analyze_cnn_sensitivity import write_csv,write_json

COLORS=['#2460a6','#d95a36','#279564','#8846ad']
STYLES=['-','--',':','-.']


def save(fig,folder,name):
    fig.savefig(folder/f'{name}.png',dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def rankings(nodes,shapley):
    result=[]
    for (context,level),rows in nodes.groupby(['context','sigma_level']):
        pred=rows.loc[rows.prediction_delta_mw.abs().idxmax()]
        ordered=rows.sort_values('grid_bus')
        isolated=ordered.iloc[np.argsort(-ordered.isolated_total_cost_delta.abs().to_numpy(),kind='stable')]
        reset=ordered.iloc[np.argsort(-ordered.reset_total_cost_effect.abs().to_numpy(),kind='stable')]
        row=dict(context=int(context),test_index=int(pred.test_index),sigma_level=float(level),
            largest_forecast_change_bus=int(pred.grid_bus),forecast_change_mw=float(pred.prediction_delta_mw),
            largest_isolated_cost_bus=int(isolated.iloc[0].grid_bus),
            isolated_top_gap=float(isolated.isolated_total_cost_delta.abs().iloc[0]-isolated.isolated_total_cost_delta.abs().iloc[1]),
            largest_reset_cost_bus=int(reset.iloc[0].grid_bus),
            reset_top_gap=float(reset.reset_total_cost_effect.abs().iloc[0]-reset.reset_total_cost_effect.abs().iloc[1]),
            forecast_matches_isolated=bool(pred.grid_bus==isolated.iloc[0].grid_bus),
            forecast_matches_reset=bool(pred.grid_bus==reset.iloc[0].grid_bus),
            isolated_sum_minus_joint=float(rows.isolated_total_cost_delta.sum()-rows.joint_cost_delta.iloc[0]),
            shapley_largest_bus=None,forecast_matches_shapley=None)
        if context==0:
            values=shapley[shapley.sigma_level==level]
            best=values.loc[values.shapley_total.abs().idxmax()]
            row.update(shapley_largest_bus=int(best.grid_bus),forecast_matches_shapley=bool(best.grid_bus==pred.grid_bus))
        result.append(row)
    return result


def paired_summary(frame):
    selected=frame[frame.role=='stratified']
    table=selected.pivot(index=['context','test_index','month','load_group','sigma_level'],columns='model',values='cost_total').reset_index()
    table['tamu_minus_direct']=table.tamu20-table.unlearned20
    table['tamu_minus_original']=table.tamu20-table.original
    table['tamu_minus_retrained']=table.tamu20-table.retrained
    summaries=[]
    for level,rows in table.groupby('sigma_level'):
        def wins(column):
            return int((rows[column]<-1e-4).sum())
        summaries.append(dict(sigma_level=float(level),samples=len(rows),
            tamu_mean_cost=float(rows.tamu20.mean()),original_mean_cost=float(rows.original.mean()),
            direct_mean_cost=float(rows.unlearned20.mean()),retrained_mean_cost=float(rows.retrained.mean()),
            tamu_mean_minus_direct=float(rows.tamu_minus_direct.mean()),
            tamu_mean_minus_original=float(rows.tamu_minus_original.mean()),
            tamu_median_minus_direct=float(rows.tamu_minus_direct.median()),
            tamu_vs_direct_wins=wins('tamu_minus_direct'),tamu_vs_original_wins=wins('tamu_minus_original'),
            tamu_vs_retrained_wins=wins('tamu_minus_retrained'),
            worst_tamu_minus_direct=float(rows.tamu_minus_direct.max()),
            best_tamu_minus_direct=float(rows.tamu_minus_direct.min())))
    return table,summaries


def figures(out,frame,nodes,shapley,paired):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    original=frame[(frame.context==0)&(frame.model=='original')].sort_values('sigma_level')
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout='constrained')
    for col,label,color in [('cost1','Stage 1',COLORS[0]),('cost2','Stage 2',COLORS[1]),('cost_total','Total',COLORS[2])]:
        ax.plot(original.sigma_level,original[col],'o-',label=label,color=color)
    ax.set(xlabel='Temperature perturbation (training SD)',ylabel='Cost (original objective units)',
           title='Original CNN: fixed test sample 0, input temperature at grid bus 1',xticks=LEVELS)
    ax.grid(alpha=.2);ax.legend()
    save(fig,folder,'01_reference_temperature_cost')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,(name,label) in enumerate(NAMES.items()):
        rows=frame[(frame.context==0)&(frame.model==name)].sort_values('sigma_level')
        for ax,column,ylabel in zip(axes,['cost_total','cost_delta'],['Absolute total cost','Change from own zero-temperature baseline']):
            ax.plot(rows.sigma_level,rows[column],marker=['o','s','x','D'][i],linestyle=STYLES[i],
                    color=COLORS[i],label=label,linewidth=1.8)
            ax.set(xlabel='Temperature perturbation (training SD)',ylabel=ylabel,xticks=LEVELS);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Four frozen heads on the same five inputs and the same actual load')
    save(fig,folder,'02_reference_four_models')
    rows=nodes[(nodes.context==0)&(nodes.sigma_level==1.)].sort_values('grid_bus')
    shap=shapley[shapley.sigma_level==1.].sort_values('grid_bus')
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    axes[0].bar(rows.grid_bus,rows.prediction_delta_mw,color=COLORS[0])
    axes[0].set(ylabel='Signed forecast change (MW)',title='Output-node prediction changes')
    axes[1].bar(rows.grid_bus-.18,rows.isolated_total_cost_delta,width=.36,label='Only this output changes',color=COLORS[1])
    axes[1].bar(rows.grid_bus+.18,rows.reset_total_cost_effect,width=.36,label='Full minus reset this output',color=COLORS[2])
    axes[1].set(ylabel='Signed effect on total cost',title='Two reoptimized counterfactual diagnostics');axes[1].legend(fontsize=7)
    axes[2].bar(shap.grid_bus,shap.shapley_total,color=COLORS[3],yerr=shap.mc_se_total,capsize=2)
    axes[2].set(ylabel='Allocated change in total cost',title='Paired permutation Shapley (mean +/- MC SE)')
    for ax in axes:
        ax.set(xlabel='Output grid bus',xticks=range(1,15));ax.axhline(0,color='0.5',linewidth=.7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Reference sample 0, +1 training SD: output-node diagnostics, not physical causal effects')
    save(fig,folder,'03_output_node_cost_attribution')
    selected=frame[frame.role=='stratified']
    fig,axes=plt.subplots(2,2,figsize=(13.8,10),layout='constrained')
    for i,(name,label) in enumerate(NAMES.items()):
        rows=selected[selected.model==name].groupby('sigma_level').cost_total.mean().reindex(LEVELS)
        axes[0,0].plot(LEVELS,rows,marker='o',linestyle=STYLES[i],color=COLORS[i],label=label)
    axes[0,0].set(title='27 distinct-date samples: mean actual cost',ylabel='Mean total cost');axes[0,0].legend(fontsize=8)
    grouped=paired.groupby('sigma_level').tamu_minus_direct
    means=grouped.mean().reindex(LEVELS)
    q25=grouped.quantile(.25).reindex(LEVELS);q75=grouped.quantile(.75).reindex(LEVELS)
    axes[0,1].plot(LEVELS,means,'o-',color=COLORS[3],label='Mean paired difference')
    axes[0,1].fill_between(LEVELS,q25,q75,alpha=.18,color=COLORS[3],label='25th-75th percentiles across dates')
    axes[0,1].axhline(0,color='0.5',linewidth=.7)
    axes[0,1].set(title='TAMU minus direct unlearning (negative = lower cost)',ylabel='Paired cost difference');axes[0,1].legend(fontsize=8)
    for column,label,color in [('tamu_minus_direct','vs direct',COLORS[1]),('tamu_minus_original','vs original',COLORS[0])]:
        wins=paired.assign(win=paired[column]<-1e-4).groupby('sigma_level').win.mean().reindex(LEVELS)*100
        axes[1,0].plot(LEVELS,wins,'o-',label=label,color=color)
    axes[1,0].set(title='Fraction of 27 samples with lower TAMU cost',ylabel='Lower-cost fraction (%)',ylim=(-2,102));axes[1,0].legend()
    table=paired.pivot_table(index=['month','load_group'],columns='sigma_level',values='tamu_minus_direct',aggfunc='mean')
    order=pd.MultiIndex.from_product([[1,2,3],['low','middle','high']])
    table=table.reindex(order).reindex(columns=LEVELS)
    bound=max(float(np.abs(table.to_numpy()).max()),1e-10)
    im=axes[1,1].imshow(table.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
    axes[1,1].set(xticks=range(5),xticklabels=[f'{x:+g}' for x in LEVELS],
        yticks=range(9),yticklabels=[f'Month {m}: {g}' for m,g in order],title='Within-month load strata (3 dates in each cell)')
    fig.colorbar(im,ax=axes[1,1],label='Mean TAMU minus direct cost')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set(xlabel='Temperature perturbation (training SD)',xticks=LEVELS);ax.grid(alpha=.2)
    axes[1,1].set_xlabel('Temperature perturbation (training SD)')
    fig.suptitle('Descriptive cohort only: January, February and first 14 days of March; one fixed model seed')
    save(fig,folder,'04_stratified_cohort_cost')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'research/cnn_cost_attribution')
    args=parser.parse_args();out=args.results_dir.resolve()
    frame=pd.read_csv(out/'results.csv');nodes=pd.read_csv(out/'node_counterfactuals.csv')
    if len(frame)!=560 or frame.context.nunique()!=28:
        raise ValueError('This report requires the full 28-context, 560-scenario study')
    shapley=pd.read_csv(out/'reference_shapley.csv')
    comparisons=rankings(nodes,shapley)
    paired,summaries=paired_summary(frame)
    write_csv(out/'node_rank_comparison.csv',comparisons)
    write_csv(out/'paired_model_costs.csv',paired.to_dict('records'))
    write_csv(out/'cohort_summary.csv',summaries)
    adverse=paired[(paired.tamu_minus_direct>1e-4)|(paired.tamu_minus_original>1e-4)].copy()
    write_csv(out/'economic_counterexamples.csv',adverse.to_dict('records'))
    manifest=json.loads((out/'manifest.json').read_text())
    summary=dict(reference_rankings=[r for r in comparisons if r['context']==0],
        total_node_rank_comparisons=len(comparisons),
        forecast_isolated_top_disagreements=sum(not r['forecast_matches_isolated'] for r in comparisons),
        forecast_reset_top_disagreements=sum(not r['forecast_matches_reset'] for r in comparisons),
        unresolved_isolated_top_gaps_le_1e_4=sum(r['isolated_top_gap']<=1e-4 for r in comparisons),
        unresolved_reset_top_gaps_le_1e_4=sum(r['reset_top_gap']<=1e-4 for r in comparisons),
        max_abs_isolated_additivity_gap=max(abs(r['isolated_sum_minus_joint']) for r in comparisons),
        shapley_efficiency_max_error=max(abs(r.shapley_total.sum()-r.joint_cost_delta.iloc[0]) for _,r in shapley.groupby('sigma_level')),
        cohort=summaries,checks=manifest['checks'])
    write_json(out/'summary.json',summary)
    figures(out,frame,nodes,shapley,paired)
    report(out,frame,nodes,shapley,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def report(out,frame,nodes,shapley,summary):
    parent=json.loads((out/'parent_provenance.json').read_text())
    checks=summary['checks']
    original=frame[(frame.context==0)&(frame.model=='original')].sort_values('sigma_level')
    original_table='\n'.join(f'| {r.sigma_level:+g} | {r.total_prediction_delta_mw:+.6f} | {r.cost1:.6f} | {r.cost2:.6f} | {r.cost_total:.6f} |' for r in original.itertuples())
    ref_ranks='\n'.join(f'| {r["sigma_level"]:+g} | {r["largest_forecast_change_bus"]} | {r["largest_isolated_cost_bus"]} | {r["largest_reset_cost_bus"]} | {r["shapley_largest_bus"]} |' for r in summary['reference_rankings'])
    cohort_table='\n'.join(f'| {r["sigma_level"]:+g} | {r["original_mean_cost"]:.6f} | {r["direct_mean_cost"]:.6f} | {r["retrained_mean_cost"]:.6f} | {r["tamu_mean_cost"]:.6f} | {r["tamu_vs_direct_wins"]}/27 | {r["tamu_vs_original_wins"]}/27 |' for r in summary['cohort'])
    paired=pd.read_csv(out/'paired_model_costs.csv')
    worst=paired.loc[paired.tamu_minus_direct.idxmax()]
    samples=json.loads((out/'selection.json').read_text())
    worst_sample=next(r for r in samples if r['test_index']==int(worst.test_index))
    plus=shapley[shapley.sigma_level==1.]
    best=plus.loc[plus.shapley_total.abs().idxmax()]
    n=summary['total_node_rank_comparisons']
    text=f'''# CNN 温度—负荷预测—运行成本：节点诊断与多日期验证

本轮沿用既有 CNN 核心与原始、20% 直接遗忘、剩余样本重拟合头、20% TAMU（L1=0.1）四种权重，种子42，helpful，case14。**没有训练模型，没有改变原成本目标、罚系数或数学约束。** 结果在独立分支 `analysis/cnn-cost-attribution`。

## 本轮回答什么

1. 保持五档温度扰动不变，实算 cost1/cost2/总 Cost，比较四种模型的经济性。
2. 检查“预测变化最大的输出节点是否也是 Cost 影响最大的输出节点”，使用两种节点反事实和抽样 Shapley，避免把预测变化量直接解释为成本贡献。
3. 在预先按日期和真实负荷选定的样本上复查结果；范围限于当前测试集。

仍扰动输入节点1的温度 `input[0,0,0,4]`，训练段7008行的 σ={parent['temperature_statistics']['train_sigma_encoded']:.12f} 编码单位，等价温差 {parent['temperature_statistics']['train_sigma_kelvin']:.9f} K。每次仅加 −1/−0.5/0/+0.5/+1σ，其他139个输入、四种权重、每个样本的真实负荷和系统约束均固定。14×10原始输入、64隐藏特征及输出缩放沿用 [上一轮结构与调用链报告](../cnn_temperature_response/README.md)。

## 1. 原始 CNN 的运行成本路径

![参考样本温度与成本](figures/01_reference_temperature_cost.png)

| 扰动/σ | 总预测变化/MW | cost1 | cost2 | 总 Cost |
| --- | --- | --- | --- | --- |
{original_table}

调用链：`temperature_cases → NN_CONV.forward → 冻结仿射头 → output_z * target_std + target_mean → Operator.solve_one(forecast_pu) → Operator.solve_two(fixed_truth_pu, PG) → cost1+cost2`。预测MW=p.u.×100。费用沿用仓库单位，没有附加市场电价或现实天气因果解释。每次求解的预测、真实负荷、七项费用、PG/LS1/LS2/GS/角度、状态和约束残差均在 [all_cost_traces.jsonl](all_cost_traces.jsonl)。

## 2. 哪个输出节点对 Cost 最重要

固定参考样本0，记零扰动预测为 f0、完整温度扰动后的预测为 f1，成本函数 C 使用同一真实负荷和原两阶段优化。

- **隔离变化：** 仅把节点 j 的预测从 f0[j] 改成 f1[j]，其他输出取 f0，计算 `C(仅j改变)−C(f0)`。
- **复位诊断：** 其他输出取完整 f1，仅将 j 复位到 f0[j]，计算 `C(f1)−C(仅j复位)`。
- **抽样 Shapley：** 用 {checks['shapley_permutations']} 条顺序（{checks['shapley_pairs']} 个独立随机顺序及其逆序对，seed42）逐一加入输出变化，平均各节点对 cost1/cost2/总 Cost 的边际影响。共享子集成本只求解一次并缓存。

这三类是**预测输出向量的反事实诊断**，不代表在14个输入节点分别改变温度，也不证明物理因果。混合输出未必能由某一真实 CNN 输入产生。总负荷预测变化不等于各节点成本贡献；隔离/复位值也不能未经验证相加。

![节点预测与Cost贡献](figures/03_output_node_cost_attribution.png)

参考样本的最大值均按绝对值排序，符号保存在原表：

| 温度扰动/σ | 最大预测变化节点 | 最大隔离Cost节点 | 最大复位Cost节点 | 最大抽样Shapley节点 |
| --- | --- | --- | --- | --- |
{ref_ranks}

在 +1σ 下，抽样 Shapley 最大的是节点 {int(best.grid_bus)}，分配的 ΔCost={best.shapley_total:+.9f}，MC标准误={best.mc_se_total:.3g}。本次参考样本中是否一致应按上表如实解释，不能预设一定不同。

所有28个样本的4个非零扰动共 {n} 次排序比较中，预测峰值与隔离Cost峰值不一致 {summary['forecast_isolated_top_disagreements']} 次，与复位Cost峰值不一致 {summary['forecast_reset_top_disagreements']} 次。隔离/复位前两名差≤1e−4的场景分别 {summary['unresolved_isolated_top_gaps_le_1e_4']}、{summary['unresolved_reset_top_gaps_le_1e_4']} 次，此时不应过度解释名次；1e−4是诊断容差，不是严格的目标值误差界。详见 [node_rank_comparison.csv](node_rank_comparison.csv)。

隔离影响求和与联合 ΔCost 的最大差为 {summary['max_abs_isolated_additivity_gap']:.9g}，不能把隔离值冒充严格的加性成本贡献。抽样 Shapley 每条路径按同一缓存成本望远镜求和，总分配与联合变化最大误差 {summary['shapley_efficiency_max_error']:.3g}。这种守恒**不意味着每个节点的 Shapley 已精确枚举**。MC标准误按独立逆序对计算，仅表示随机顺序抽样误差，不包含求解误差或数据样本不确定性；小于数值求解误差的标准误不能作为有效小数位或总体置信区间。

原始逐节点数据：[node_counterfactuals.csv](node_counterfactuals.csv)、[reference_shapley.csv](reference_shapley.csv)、[coalition_costs.csv](coalition_costs.csv)。已保存完整顺序和逐路径边际值，可核验分摊。

## 3. 相同输入下四种模型的经济性

![参考样本四模型](figures/02_reference_four_models.png)

绝对 Cost 回答哪个模型在同一真实负荷下成本较低；相对自身零扰动的 ΔCost 回答模型响应幅度。两者含义不同，响应幅度小不自动等于经济性好。重新训练仍指原项目协议下的910参数预测头重新拟合，核心不动。TAMU权重使用原测试平均Cost分数恢复，未针对本次温度场景重新优化。

## 4. 不同日期与真实负荷水平

在任何本轮推理/Cost之前，按当前测试集的1月、2月、3月分别计算真实总负荷三分位。每个月的低/中/高层内，取最接近10%、50%、90%负荷分位的小时，保留互不重复的日期。形成 **27个分层样本、27个日期**，另保留原参考样本0，总计28个日期。原参考点不计入下表分层平均值，防止额外加权。

选样只用日期与真实负荷，不看原始/TAMU的预测或Cost。选择规则、三分位阈值和哈希在推理前保存；见 [selection.json](selection.json)、[parent_provenance.json](parent_provenance.json)。低/中/高是各月内部相对水平，不能解释为全年的绝对低/高负荷。

![分层日期成本比较](figures/04_stratified_cohort_cost.png)

| 扰动/σ | 原始平均Cost | 直接遗忘 | 重拟合头 | TAMU | TAMU低于直接遗忘 | TAMU低于原始 |
| --- | --- | --- | --- | --- | --- | --- |
{cohort_table}

低于以差值<−1e−4判定。全部样本逐一配对比较，没有跨日期混配。图中阴影是所选27个日期的四分位区间，**不是总体置信区间**。逐场景结果/MSE/MAPE见 [results.csv](results.csv)，14节点预测见 [predictions_14_buses.csv](predictions_14_buses.csv)，配对差值见 [paired_model_costs.csv](paired_model_costs.csv)，分层汇总见 [cohort_summary.csv](cohort_summary.csv)。

所选27个日期中，TAMU在五档扰动下的平均Cost均较原始/直接遗忘低，但不逐样本保证。实际反例：test index {int(worst.test_index)}（{worst_sample['date']}、源Hour {worst_sample['source_hour']}，月内{worst_sample['load_group']}负荷），{worst.sigma_level:+g}σ 时 TAMU={worst.tamu20:.6f}，直接遗忘={worst.unlearned20:.6f}，TAMU反而高 {worst.tamu_minus_direct:.6f}。所有较原始或直接遗忘更贵的场景见 [economic_counterexamples.csv](economic_counterexamples.csv)。

## 5. 验证与复算

- 单参考样本最小真实测试先通过：四模型×五档、节点反事实和8条顺序，共960次QP；正式实验使用128条顺序。
- 正式实验 {checks['scenario_rows']} 个模型场景、{checks['forecast_values']} 个预测数值，节点隔离/复位表 {checks['node_counterfactual_rows']} 行，共 {checks['qp_solves']} 次QP。全部状态为 {checks['statuses']}；实际求解器 {checks['solvers']}；最大约束残差 {checks['max_constraint_violation']:.3g}，七项费用分解最大误差 {checks['max_objective_error']:.3g}。
- 所有模型权重哈希不变、真实负荷逐次检查、零扰动预测逐位一致、零扰动最大Cost差 {checks['zero_cost_max_error']:.3g}。自动化测试与完整重放核验记录见 [validation.json](validation.json)、[test-results.txt](test-results.txt)。
- 原始数学源码、旧实验脚本与历史结果均保留。新增 [cost_attribution.py](../../experiments/cost_attribution.py)、[report_cost_attribution.py](../../experiments/report_cost_attribution.py)、[validate_cost_attribution.py](../../experiments/validate_cost_attribution.py) 及测试；分支未合并 master。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 已有年度预处理数据及上一轮快照时运行
.venv/bin/python experiments/cost_attribution.py --smoke --output-dir /tmp/cost-path-smoke-new
.venv/bin/python experiments/cost_attribution.py --output-dir /tmp/cost-path-full-new
# 使用本目录保存的真实28条样本、训练σ及冻结权重复算，无需训练或全年原数据
.venv/bin/python experiments/cost_attribution.py --replay-from research/cnn_cost_attribution --output-dir /tmp/cost-path-replay-new
.venv/bin/python experiments/report_cost_attribution.py
.venv/bin/python experiments/validate_cost_attribution.py --replay-dir /tmp/cost-path-replay-new
.venv/bin/python -m pytest -q
```

输出目录必须为空。依赖沿用现有 [requirements.txt](../../requirements.txt)，CPU环境和求解器限制见 [成本修复报告](../../COST_REPRODUCTION_REPORT.md)。

## 结论范围

本轮给出指定温度输入→14个输出预测→两阶段Cost的实际路径，并分开核验绝对经济性、响应幅度和输出节点成本诊断。不同节点与模型的结论以实测表为准，不预设TAMU在所有场景更优，也不预设预测峰值和成本峰值必然不同。

虽然新增多日期样本，**测试期仍只有2019-01-01至03-14**，尚未覆盖夏季/全年；只有seed42和一个输入节点/特征。分层样本不是随机总体抽样；同一个训练模型的观测可能相关。原全年归一化先于划分、helpful/TAMU使用测试性能信息的原协议局限仍存在。未增加ERCOT数据或改变节点映射。

[下载完整结果ZIP](https://github.com/syh-Mason/task_aware_machine_unlearning/blob/analysis/cnn-cost-attribution/exports/cnn_cost_attribution.zip?raw=true)。
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    brief=f'''# 导师简报：温度输入如何影响预测与运行成本

已复用既有CNN和四种预测头，完成原五档扰动、两阶段成本、节点诊断及多日期验证。没有重新训练或修改原优化问题。

**参考样本：** 输入节点1温度改变±0.5/±1训练σ；Cost随CNN输出送入原调度/再调度求解。下表为原始CNN：

| 扰动/σ | 总预测变化/MW | cost1 | cost2 | 总Cost |
| --- | --- | --- | --- | --- |
{original_table}

**最大预测变化是否等于最大成本影响？** 采用只改一个输出、从完整向量复位一个输出、128条顺序抽样Shapley三种诊断，而非直接拿MW当贡献。

| 扰动/σ | 最大预测节点 | 最大隔离Cost节点 | 最大复位Cost节点 | 最大Shapley节点 |
| --- | --- | --- | --- | --- |
{ref_ranks}

跨28个日期的112次非零扰动比较，预测峰值与隔离/复位成本峰值不一致次数分别为 {summary['forecast_isolated_top_disagreements']}/{summary['forecast_reset_top_disagreements']}。参考样本+1σ下，节点 {int(best.grid_bus)} 的抽样Shapley为 {best.shapley_total:+.6f}。这些是混合预测向量的成本诊断，不是物理因果结论。

**TAMU是否保持更好经济性？** 另选27个不同日期，按1/2/3月×月内低/中/高真实负荷分层，每层3例，选样不看预测或成本：

| 扰动/σ | 原始均Cost | 直接遗忘 | 重拟合头 | TAMU | TAMU低于直接 | TAMU低于原始 |
| --- | --- | --- | --- | --- | --- | --- |
{cohort_table}

绝对成本与相对自身基准的响应幅度分开解释。单个正向扰动更稳，不意味着所有方向或所有日期都更经济。

具体反例：{worst_sample['date']}、test {int(worst.test_index)}、{worst.sigma_level:+g}σ 时，TAMU比直接遗忘贵 {worst.tamu_minus_direct:.6f}，因此不能表述为所有场景均更优。

**验证：** {checks['scenario_rows']}组模型场景，{checks['qp_solves']}次原QP，全部optimal；零扰动恢复，模型权重/每次真实负荷固定，费用完整。

**局限：** 当前测试期只有1月到3月14日；仅seed42和节点1温度。Shapley是顺序抽样，MC误差不等于样本或求解不确定性。原数据归一化及测试分数选样限制未消除。不能推广至全年/夏季/ERCOT。

交付：[完整报告](README.md)、[四张图](figures/)、[节点诊断](node_counterfactuals.csv)、[配对成本](paired_model_costs.csv)、[可复算脚本](../../experiments/cost_attribution.py)。
'''
    (out/'导师汇报.md').write_text(brief,encoding='utf-8')


if __name__=='__main__':
    main()

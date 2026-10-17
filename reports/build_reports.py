"""Build two experiment reports and a group-meeting deck from frozen evidence.

Uses the document toolchain, not the project venv. No experiments or model fits
are executed. All reported numeric tables are populated from committed CSV/JSON.
"""
from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches as PInches, Pt as PPt
from pptx.dml.color import RGBColor as PColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/cnn_experiment_reports'
ASSETS=OUT/'assets'
E1=ROOT/'research/cnn_temperature_response'
E2=ROOT/'research/cnn_cost_attribution'
STAMP='2026-10-17'
FONT='Noto Sans CJK SC'
NAVY='19354B';TEAL='167D8D';GRAY='526675';LIGHT='EAF3F4';RED='B53D3D'
MODEL_NAMES={'original':'原始 CNN','unlearned20':'直接遗忘20%',
             'retrained':'剩余样本重拟合头','tamu20':'TAMU（L1=0.1）'}
COLORS=['#2460A6','#D65B35','#269261','#8846AD']
STYLES=['-','--',':','-.']
LEVELS=[-1.,-.5,0.,.5,1.]


def read_json(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def f(value,digits=6):return f'{float(value):.{digits}f}'
def signed(value,digits=6):return f'{float(value):+.{digits}f}'


def load():
    a=pd.read_csv(E1/'results.csv')
    b=pd.read_csv(E2/'results.csv')
    pred=pd.read_csv(E1/'predictions_14_buses.csv')
    paired=pd.read_csv(E2/'paired_model_costs.csv')
    s1=read_json(E1/'summary.json');s2=read_json(E2/'summary.json')
    v1=read_json(E1/'validation.json');v2=read_json(E2/'validation.json')
    m1=read_json(E1/'manifest.json')
    assert len(a)==20 and len(b)==560 and len(paired)==135
    assert v1['tests_passed']==49 and v2['tests_passed']==57
    assert v1['replay_verified'] and v2['replay_verified']
    assert v1['tests_failed']==v2['tests_failed']==0
    nonzero=paired[paired.sigma_level!=0]
    assert len(nonzero)==108 and (nonzero.tamu_minus_direct < -1e-4).sum()==106
    return dict(a=a,b=b,pred=pred,paired=paired,s1=s1,s2=s2,v1=v1,v2=v2,m1=m1,
        cohort=pd.read_csv(E2/'cohort_summary.csv'),
        shap=pd.read_csv(E2/'reference_shapley.csv'),
        ranks=pd.read_csv(E2/'node_rank_comparison.csv'),
        selection=read_json(E2/'selection.json'),
        traces=read_json(E1/'cost_traces.json'))


def setup_plots():
    path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(path)
    name=font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({'font.family':name,'font.size':15,'axes.labelsize':16,
        'axes.titlesize':17,'legend.fontsize':13,'xtick.labelsize':14,'ytick.labelsize':14,
        'axes.unicode_minus':False,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def save_fig(fig,name):
    for extension in ['png','pdf']:
        fig.savefig(ASSETS/f'{name}.{extension}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def draw_charts(d):
    setup_plots()
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for bus in range(1,15):
        rows=d['pred'][(d['pred'].model=='original')&(d['pred'].grid_bus==bus)].sort_values('sigma_level')
        ax.plot(rows.sigma_level,rows.delta_mw,'o-',lw=1.6,ms=4,label=f'节点{bus:02d}',color=plt.get_cmap('tab20')(bus-1))
    ax.set(xlabel='温度扰动（训练集标准差 σ）',ylabel='节点预测负荷变化 / MW',xticks=LEVELS)
    ax.axhline(0,color='0.5',lw=.7);ax.grid(alpha=.18);ax.legend(ncol=2,fontsize=11,loc='center left',bbox_to_anchor=(1,.5))
    save_fig(fig,'r1_prediction_five_levels')
    fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
    rows=d['a'][d['a'].model=='original'].sort_values('sigma_level')
    for col,label,color in [('cost1','阶段一 cost1',COLORS[0]),('cost2','阶段二 cost2',COLORS[1]),('cost_total','总 Cost',COLORS[2])]:
        ax.plot(rows.sigma_level,rows[col],'o-',label=label,color=color,lw=2)
    ax.set(xlabel='温度扰动（训练集标准差 σ）',ylabel='成本（原仓库目标函数单位）',xticks=LEVELS)
    ax.grid(alpha=.18);ax.legend(loc='center left',bbox_to_anchor=(1,.5),fontsize=12)
    save_fig(fig,'r1_two_stage_cost')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for i,(name,label) in enumerate(MODEL_NAMES.items()):
        rows=d['a'][d['a'].model==name].sort_values('sigma_level')
        for ax,col,title in zip(axes,['cost_total','cost_delta'],['绝对总成本','相对各自零扰动的成本变化']):
            ax.plot(rows.sigma_level,rows[col],marker=['o','s','x','D'][i],linestyle=STYLES[i],color=COLORS[i],label=label,lw=2)
            ax.set(title=title,xlabel='温度扰动 / σ',xticks=LEVELS);ax.grid(alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=12)
    save_fig(fig,'r1_four_model_cost')
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    rows=d['pred'][(d['pred'].model=='original')&(d['pred'].sigma_level==1)].sort_values('grid_bus')
    colors=[TEAL if bus!=4 else RED for bus in rows.grid_bus]
    ax.bar(rows.grid_bus,rows.delta_mw,color=['#'+c for c in colors])
    ax.set(xlabel='输出电网节点',ylabel='预测负荷变化 / MW',xticks=range(1,15))
    ax.axhline(0,color='0.5',lw=.7);ax.grid(axis='y',alpha=.18)
    save_fig(fig,'r1_plus1_node_changes')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    shap=d['shap'][d['shap'].sigma_level==1].sort_values('grid_bus')
    axes[0].bar(rows.grid_bus,rows.delta_mw,color=COLORS[0]);axes[0].set(title='CNN 输出变化',ylabel='预测变化 / MW')
    axes[1].bar(shap.grid_bus,shap.shapley_total,color=COLORS[3]);axes[1].set(title='抽样 Shapley 成本分摊',ylabel='分摊的 ΔCost')
    for ax in axes:
        ax.set(xlabel='输出电网节点',xticks=range(1,15));ax.axhline(0,color='0.5',lw=.7);ax.grid(axis='y',alpha=.18)
    save_fig(fig,'r2_node_attribution')
    fig,axes=plt.subplots(1,2,figsize=(12,5.0),layout='constrained')
    selected=d['b'][d['b'].role=='stratified']
    for i,(name,label) in enumerate(MODEL_NAMES.items()):
        mean=selected[selected.model==name].groupby('sigma_level').cost_total.mean().reindex(LEVELS)
        axes[0].plot(LEVELS,mean,'o',linestyle=STYLES[i],color=COLORS[i],lw=2,label=label)
    axes[0].set(title='27 个分层日期：平均成本',ylabel='总 Cost')
    grouped=d['paired'].groupby('sigma_level').tamu_minus_direct
    axes[1].plot(LEVELS,grouped.mean().reindex(LEVELS),'o-',color=COLORS[3],lw=2)
    axes[1].fill_between(LEVELS,grouped.quantile(.25).reindex(LEVELS),grouped.quantile(.75).reindex(LEVELS),color=COLORS[3],alpha=.16)
    axes[1].axhline(0,color='0.5',lw=.7)
    axes[1].set(title='配对成本差：TAMU − 直接遗忘',ylabel='均值与日期间 25%—75% 分位区间')
    for ax in axes:
        ax.set(xlabel='温度扰动 / σ',xticks=LEVELS);ax.grid(alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=12)
    save_fig(fig,'r2_cohort_mean_cost')
    heat=d['paired'].pivot_table(index=['month','load_group'],columns='sigma_level',values='tamu_minus_direct',aggfunc='mean')
    order=pd.MultiIndex.from_product([[1,2,3],['low','middle','high']]);heat=heat.reindex(order).reindex(columns=LEVELS)
    fig,ax=plt.subplots(figsize=(9,5.6),layout='constrained');bound=np.abs(heat.to_numpy()).max()
    im=ax.imshow(heat,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
    groups={'low':'低','middle':'中','high':'高'}
    ax.set(yticks=range(9),yticklabels=[f'{m}月 / {groups[g]}负荷' for m,g in order],
        xticks=range(5),xticklabels=[f'{x:+g}' for x in LEVELS],xlabel='温度扰动 / σ')
    fig.colorbar(im,ax=ax,label='层内平均成本差：TAMU − 直接遗忘')
    save_fig(fig,'r2_month_load_strata')
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for i,(name,label) in enumerate(MODEL_NAMES.items()):
        rows=d['b'][(d['b'].test_index==1686)&(d['b'].model==name)].sort_values('sigma_level')
        ax.plot(rows.sigma_level,rows.cost_total,marker=['o','s','x','D'][i],linestyle=STYLES[i],color=COLORS[i],lw=2,label=label)
    ax.set(xlabel='温度扰动 / σ',ylabel='总 Cost',xticks=LEVELS)
    ax.grid(alpha=.18);ax.legend(loc='center left',bbox_to_anchor=(1,.5),fontsize=12)
    save_fig(fig,'r2_counterexample')
    # A compact diagram is drawn as vector shapes; all values reflect the actual code path.
    fig,ax=plt.subplots(figsize=(12,3),layout='constrained');ax.axis('off')
    boxes=[('原始输入','14节点 × 10特征'),('冻结 CNN','卷积 → 64维 tanh'),('预测头','64 → 14维输出'),('反标准化','z × std + mean'),('阶段一','预测负荷\nPG 与 cost1'),('阶段二','真实负荷 + PG\ncost2')]
    for i,(title,sub) in enumerate(boxes):
        x=.012+i*.165
        ax.add_patch(FancyBboxPatch((x,.31),.145,.48,boxstyle='round,pad=.004',
            facecolor='#EAF3F4',edgecolor='#167D8D',transform=ax.transAxes))
        ax.text(x+.0725,.68,title,ha='center',va='center',fontsize=13,fontweight='bold',transform=ax.transAxes)
        ax.text(x+.0725,.48,sub,ha='center',va='center',fontsize=10.5,transform=ax.transAxes)
        if i<5:ax.annotate('',xy=(x+.161,.55),xytext=(x+.15,.55),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#19354B',lw=1.3))
    ax.text(.5,.12,'仅改变输入节点1的温度；固定权重、其他输入和真实负荷；总 Cost = cost1 + cost2',ha='center',fontsize=14,transform=ax.transAxes)
    save_fig(fig,'shared_influence_path')


def set_run_font(run,size=None,bold=None,color=None):
    run.font.name=FONT
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),FONT)
    if size:run.font.size=Pt(size)
    if bold is not None:run.bold=bold
    if color:run.font.color.rgb=RGBColor.from_string(color)


class Report:
    def __init__(self,folder,title,subtitle):
        self.folder=OUT/folder;self.folder.mkdir(parents=True,exist_ok=True)
        self.doc=Document();self.md=[];self.pages=0
        section=self.doc.sections[0]
        section.page_width=Inches(8.2677);section.page_height=Inches(11.6929)
        section.left_margin=section.right_margin=Inches(.72)
        section.top_margin=Inches(.66);section.bottom_margin=Inches(.63)
        section.header_distance=section.footer_distance=Inches(.28)
        for name in ['Normal','Body Text','Caption','Heading 1','Heading 2']:
            style=self.doc.styles[name];style.font.name=FONT
            style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),FONT)
        normal=self.doc.styles['Normal'];normal.font.size=Pt(10.5)
        normal.paragraph_format.line_spacing=1.22;normal.paragraph_format.space_after=Pt(6)
        for name,size in [('Heading 1',17),('Heading 2',12.5)]:
            style=self.doc.styles[name];style.font.size=Pt(size);style.font.color.rgb=RGBColor.from_string(NAVY)
            style.paragraph_format.space_before=Pt(3);style.paragraph_format.space_after=Pt(8)
        self.doc.styles['Caption'].font.size=Pt(9)
        h=section.header.paragraphs[0];h.alignment=WD_ALIGN_PARAGRAPH.RIGHT
        set_run_font(h.add_run('CNN 输入扰动—负荷预测—运行成本'),8.5,color=GRAY)
        footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(footer.add_run('研究记录  |  '),8,color=GRAY)
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
        self.title=title;self.subtitle=subtitle
        self.doc.core_properties.title=title;self.doc.core_properties.subject=subtitle
        self.doc.core_properties.author='syh-Mason/task_aware_machine_unlearning 项目记录'
    def page(self,title):
        if self.pages:self.doc.add_page_break()
        self.pages+=1
        self.doc.add_heading(title,level=1)
        self.md.extend(['',f'## {title}',''])
    def p(self,text,bold=False):
        p=self.doc.add_paragraph();r=p.add_run(text);set_run_font(r,bold=bold)
        self.md.extend([text,''])
    def bullet(self,text):
        p=self.doc.add_paragraph(style='List Bullet');set_run_font(p.add_run(text))
        self.md.append('- '+text)
    def h(self,text):
        self.doc.add_heading(text,level=2);self.md.extend(['',f'### {text}',''])
    def table(self,headers,rows,size=9.2,widths=None):
        table=self.doc.add_table(rows=1,cols=len(headers));table.autofit=False
        if widths is None:widths=[1]*len(headers)
        for col,w in zip(table.columns,widths):col.width=Inches(6.82*w/sum(widths))
        for cell,w in zip(table.rows[0].cells,widths):cell.width=Inches(6.82*w/sum(widths))
        for i,text in enumerate(headers):
            cell=table.rows[0].cells[i];cell.text=str(text)
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),NAVY);cell._tc.get_or_add_tcPr().append(shade)
        repeat=OxmlElement('w:tblHeader');repeat.set(qn('w:val'),'true');table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for j,values in enumerate(rows):
            cells=table.add_row().cells
            for cell,w,text in zip(cells,widths,values):
                cell.width=Inches(6.82*w/sum(widths));cell.text=str(text)
                if j%2==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F0F5F6');cell._tc.get_or_add_tcPr().append(shade)
        for i,row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=p.paragraph_format.space_before=Pt(2.5)
                    p.paragraph_format.line_spacing=1.04
                    for r in p.runs:set_run_font(r,size=size,bold=i==0,color='FFFFFF' if i==0 else NAVY)
        self.doc.add_paragraph().paragraph_format.space_after=Pt(0)
        self.md.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |'])
        self.md.extend('| '+' | '.join(map(str,row))+' |' for row in rows);self.md.append('')
    def figure(self,name,caption,width=6.65):
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(1)
        p.add_run().add_picture(str(ASSETS/f'{name}.png'),width=Inches(width))
        cap=self.doc.add_paragraph(caption,style='Caption');cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
        self.md.extend([f'![{caption}](../assets/{name}.png)',''])
    def code(self,text):
        p=self.doc.add_paragraph();p.paragraph_format.space_after=Pt(5)
        r=p.add_run(text);set_run_font(r,size=8.2,color=GRAY)
        r.font.name='DejaVu Sans Mono';self.md.extend(['```text',text,'```',''])
    def save(self,name):
        self.doc.save(self.folder/f'{name}.docx')
        (self.folder/f'{name}.md').write_text('# '+self.title+'\n\n'+self.subtitle+'\n'+'\n'.join(self.md),encoding='utf-8')


def report_one(d):
    r=Report('01_single_sample','实验报告一：CNN 单节点温度扰动的预测与成本影响','单样本、五档扰动与20%遗忘前后对照')
    a=d['a'];original=a[a.model=='original'].sort_values('sigma_level')
    sigma=d['m1']['temperature_statistics'];v=d['v1']
    r.page('实验报告一｜单节点温度扰动')
    r.p('CNN 单输入特征扰动—负荷预测—电网运行成本',bold=True)
    r.p('实验类型：固定模型的数值干预与两阶段运行成本评估。整理日期：'+STAMP+'。本报告汇编既有实测记录，没有重新训练模型或重新生成实验结果。')
    r.table(['记录项','说明'],[
        ['结果版本','analysis/cnn-temperature-response；提交 b5fd9f1'],
        ['数据与系统','官方 TX-123BT 2019 合成测试系统数据；case14'],
        ['固定条件','seed=42，helpful，删除20%敏感样本；TAMU 平均L1=0.1'],
        ['参考样本','test index 0，源 Day 1 / Hour 1；不推断数据时区'],
        ['实验规模','4模型 × 5档 = 20场景；280个逐节点预测；48次QP'],
        ['文档定位','本报告只讨论单样本；多日期验证见实验报告二']],widths=[1,3])
    r.h('摘要')
    r.p('以已有CNN为基础，仅改变输入电网节点1的温度，幅度为训练集标准差的−1、−0.5、0、+0.5、+1倍。其他输入、模型权重及该样本真实负荷不变。实验贯通了输入编码、CNN前向传播、14节点负荷预测、阶段一调度和阶段二再调度的完整路径。')
    r.p('在原始CNN上，+1σ使总预测负荷变化−0.827542 MW，总Cost变化−6.778363；最大输出变化为节点4的−0.278248 MW。直接遗忘与剩余样本重拟合预测头的响应近乎重叠。TAMU在+1σ方向的Cost响应幅度较小，在−1σ方向反而更大，因此不能认定它全面降低敏感性。')
    r.h('本轮可支持的结论')
    r.bullet('一个温度输入会通过CNN的隐藏表示影响多个输出节点，并进一步改变两阶段运行成本。')
    r.bullet('绝对经济性与相对自身基准的响应幅度必须分别报告。')
    r.bullet('单点干预不代表真实天气事件，也不构成温度对实际负荷的因果证据。')

    r.page('1  数据、输入结构与标准化')
    r.p('原始CNN输入为(batch,1,14,10)，按预处理bus文件编号的数值顺序排列。140个原始编码值进入卷积网络，64维是学习到的tanh隐藏表示，不是64种物理特征。Load列是预测目标，不属于输入。')
    features=[('0','星期正弦','周期编码'),('1','星期余弦','周期编码'),('2','小时正弦','周期编码'),('3','小时余弦','周期编码'),
        ('4','温度 / 原单位K','节点全年z-score'),('5','短波辐射 / W·m⁻²','节点全年z-score'),('6','长波辐射 / W·m⁻²','节点全年z-score'),
        ('7','纬向风速 / m·s⁻¹','节点全年z-score'),('8','经向风速 / m·s⁻¹','节点全年z-score'),('9','风速 / m·s⁻¹','节点全年z-score')]
    r.table(['节点内零基索引','输入特征','模型所见表示'],features,size=9.5,widths=[1,2,1.6])
    r.table(['关键核验项','实际值'],[
        ['本次温度位置','input[0,0,0,4]；140维展开零基索引4'],
        ['节点对应','电网节点1 → bus_2.csv → 原TX零基索引2 / 标签3'],
        ['训练/测试划分','训练7008行：全年行1752..8759；测试1752行：全年行0..1751'],
        ['训练σ（ddof=0）',f'{sigma["train_sigma_encoded"]:.12f} 编码单位；{sigma["train_sigma_kelvin"]:.9f} K'],
        ['原全年统计（ddof=1）',f'均值{sigma["original_annual_mean_kelvin"]:.9f} K；标准差{sigma["original_annual_sample_std_kelvin"]:.9f} K'],
        ['参考温度',f'{sigma["reference_temperature_kelvin"]:.9f} K；float32编码{sigma["reference_encoded_float32"]:.12f}']],size=9,widths=[1,3])
    r.p('训练σ仅由原训练段计算；原模型使用的全年归一化保持不变。选中节点8760小时原温度已与365个官方工作簿及预处理CSV逐小时核对。原项目先全年归一化再划分数据的限制仍存在。')

    r.page('2  方法：从输入到运行成本')
    r.figure('shared_influence_path','图1  真实代码对应的输入—预测—运行成本路径')
    r.p('CNN核心为两层卷积与ReLU，展平为280维，经线性层和tanh得到64维特征；再由仿射预测头输出14维标准化负荷。目标反标准化后以p.u.进入运行优化，MW仅用于结果展示。')
    r.code('x_temperature(k) = x_temperature(0) + k * sigma_train\nk in {-1, -0.5, 0, +0.5, +1}\nforecast_pu = output_z * target_std + target_mean\nforecast_MW = forecast_pu * 100')
    r.h('保留原始两阶段目标函数')
    r.p('阶段一：C1 = PGᵀ diag(q) PG + cᵀ PG + λ ΣLS1。阶段二：C2 = λ ΣLS2 + κ ΣLS2² + ρ ΣGS + η ΣGS²。总Cost=C1+C2。参数λ=3500、κ=1250、ρ=700、η=250。Cost沿用仓库目标函数单位，不解释为实际市场电价或美元/小时。')
    r.p('阶段一使用预测负荷；阶段二固定该样本真实负荷，并使用阶段一的PG。发电上下限、线路双向限额、功率平衡、LS/GS非负与参考角约束全部沿用原实现，未新增LS≤负荷或GS≤PG等约束。负预测沿用原逻辑截为0。')
    r.table(['模型','本轮使用方式'],[[MODEL_NAMES[n],s] for n,s in [
        ('original','敏感集拟合的原始预测头'),('unlearned20','20% helpful直接遗忘后的预测头'),
        ('retrained','剩余3925条敏感样本上重新拟合的预测头'),('tamu20','按Cost分数、平均L1=0.1重加权恢复的预测头')]],size=9,widths=[1.4,3])
    r.p('四模型共享同一个冻结CNN核心。“重新训练CNN”在这里仅指原项目协议下910个仿射参数的重新拟合，不是从头训练整个卷积网络。')

    r.page('3  原始CNN的负荷预测响应')
    r.figure('r1_prediction_five_levels','图2  五档温度扰动引起的14个输出节点变化')
    r.table(['扰动/σ','总预测变化/MW','MSE/p.u.²','MAPE/%'],[
        [f'{q.sigma_level:+g}',signed(q.forecast_total_delta_mw),f(q.mse_pu2,9),f(q.mape_percent)] for q in original.itertuples()],size=9.5)
    r.p('MSE和MAPE均对同一条样本的14个输出节点计算；它们不是全测试集平均指标。所有扰动保持真实负荷不变，仅评价预测器对编码温度变化的响应。')
    r.p('+1σ时，全部14个输出节点变化超过1e−6 MW，最大绝对变化出现在节点4，为−0.278248 MW。负荷总量的变化为−0.827542 MW。曲线在正、负扰动方向呈非对称响应，不宜以单一线性斜率概括。')

    r.page('4  两阶段成本与费用分解')
    r.figure('r1_two_stage_cost','图3  原始CNN五档预测对应的运行成本')
    r.table(['扰动/σ','cost1','cost2','总Cost'],[[f'{q.sigma_level:+g}',f(q.cost1),f(q.cost2),f(q.cost_total)] for q in original.itertuples()],size=9.4)
    r.p('+1σ相对零扰动：cost1变化−0.481661，cost2变化−6.296702，总Cost变化−6.778363。主要变化来自阶段二GS过量发电处理费用；LS1和LS2的变化接近数值求解误差。该费用解释仅针对本参考样本。')
    traces=d['traces']['original'];zero=traces[2];plus=traces[4]
    names={'dispatch_generation_quadratic':'发电二次费用','dispatch_generation_linear':'发电一次费用','dispatch_shedding':'阶段一削减费用',
        'redispatch_shedding_linear':'阶段二削减一次费用','redispatch_shedding_quadratic':'阶段二削减二次费用',
        'redispatch_storage_linear':'GS一次费用','redispatch_storage_quadratic':'GS二次费用'}
    r.table(['原目标费用项','+1σ相对零扰动变化'],[[label,
        signed(plus['components'][key]-zero['components'][key]) if abs(plus['components'][key]-zero['components'][key])>=1e-6 else '约0（<1e−6）'] for key,label in names.items()],size=9,widths=[2,1])

    r.page('5  遗忘前后四种模型的响应')
    r.figure('r1_four_model_cost','图4  绝对成本和相对自身基准响应的对照')
    rows=[]
    for name in MODEL_NAMES:
        values=a[a.model==name].set_index('sigma_level');z=values.loc[0.];p=values.loc[1.];n=values.loc[-1.]
        rows.append([MODEL_NAMES[name],f(z.cost_total),signed(p.forecast_total_delta_mw),signed(p.cost_delta),signed(n.cost_delta)])
    r.table(['模型','零扰动Cost','+1σ Δ预测/MW','+1σ ΔCost','−1σ ΔCost'],rows,size=8.8,widths=[1.5,1,1,1,1])
    r.p('直接遗忘与重拟合头的五档总Cost最大差异为0.000472933，二者曲线近乎重叠。TAMU的零扰动绝对成本较低，+1σ时成本响应幅度也较小；但−1σ的响应幅度更大。这两种事实不矛盾：绝对成本与响应幅度衡量不同问题。')
    r.p('本实验没有为每档温度扰动重新优化TAMU权重。TAMU针对原测试平均Cost的一阶分数进行重加权，不提供逐场景成本或输入敏感性都下降的保证。')

    r.page('6  恢复核验、求解状态与复现')
    r.table(['验收项','记录结果'],[
        ['模型恢复','前三种预测头及CNN核心与上一轮保存参数逐位相同'],
        ['TAMU恢复','全测试MSE=0.004006459731、MAPE=9.124598238%、平均Cost=470.887519328，与历史L1=0.1记录一致'],
        ['加权CG','info=0；独立加权正规方程相对残差9.2462e−6 < 1e−5'],
        ['零扰动','预测逐位恢复，Cost差为0'],
        ['正式规模','20组模型场景；另有4次独立基准Cost，合计48个QP'],
        ['求解质量',f'全部optimal；最大约束残差{v["max_constraint_violation"]:.3g}'],
        ['回归测试','该轮49项通过，0失败、0跳过；后来扩展为累计57项'],
        ['重放','280个预测逐位相同；成本表和求解轨迹逐字节相同']],size=9.3,widths=[1,3.7])
    r.p('3925变量的TAMU重加权LP受GUROBI内置许可证规模限制，按原有自动回退机制由ECOS求解同一LP；没有替换目标或约束。本轮小型运行QP使用GUROBI。调试中修正了新增核验脚本的权重/偏置展平顺序，并统一单样本前向，避免批量矩阵运算舍入差异被当作扰动效应。失败输出未用于最终结果。')
    r.h('复算入口（仓库根目录，输出目录须为空）')
    r.code('.venv/bin/python experiments/input_sensitivity.py \\\n  --replay-from research/cnn_temperature_response \\\n  --output-dir /tmp/temperature-report-replay')
    r.code('.venv/bin/python experiments/validate_temperature_response.py \\\n  --replay-dir /tmp/temperature-report-replay')
    r.p('数值记录来自既有实验validation.json与测试日志，本次编制报告没有重新运行上述实验。复算仍需原项目固定版本依赖及兼容求解器。')

    r.page('7  结论、局限与源码索引')
    r.h('结论')
    r.p('已完成指定温度坐标到14节点预测、两阶段决策及完整Cost的可追踪实验；直接遗忘与重拟合头响应接近，TAMU表现出方向相关的响应变化。该结果为下一轮节点成本诊断和多日期验证提供了固定模型、真实输入及可重放基准。')
    r.h('必须保留的解释边界')
    for text in ['只有一个样本、一个输入节点、温度一个特征和seed42，不支持全年或跨系统结论。',
        '输入编码的人工加法扰动保持真实负荷不变，不是实际发生的天气事件，也不证明气温对实际负荷的因果关系。',
        '训练σ只用训练段，但原全年天气与目标归一化先于训练/测试划分；helpful与TAMU还使用测试性能分数。新增σ控制没有消除这些原协议局限。',
        '成本是项目目标函数数值；没有使用真实市场电价，也没有在本实验中接入ERCOT数据。']:
        r.bullet(text)
    r.table(['源码/产物','用途'],[
        ['clean_data.py / utils/dataset.py','天气行映射、特征顺序、数据排列、标准化与划分'],
        ['utils/net.py: NN_CONV.forward','14×10输入 → 64维隐藏表示'],
        ['func_operation.py / utils/funcs.py','预测头恢复、预测反标准化和评价'],
        ['utils/optimization.py','原两阶段调度、求解状态、变量与约束'],
        ['experiments/input_sensitivity.py','独立扰动与完整结果保存'],
        ['research/cnn_temperature_response/','结果CSV、280个预测、费用轨迹、快照、验证记录']],size=9,widths=[1.5,2.5])
    r.p('数据与模型来源：task_aware_machine_unlearning 原项目；Xu与Teng，Task-Aware Machine Unlearning and Its Application in Load Forecasting，IEEE TPWRS，DOI:10.1109/TPWRS.2024.3376828。TX-123BT数据的原始引用及官方归档信息见仓库COST_REPRODUCTION_REPORT.md与readme.md。')

    r.page('附录A  全部20组指标与数据索引')
    abbreviations={'original':'原始','unlearned20':'直接','retrained':'重拟合','tamu20':'TAMU'}
    r.table(['模型','σ','MSE/p.u.²','MAPE/%','cost1','cost2','总Cost'],[
        [abbreviations[q.model],f'{q.sigma_level:+g}',f(q.mse_pu2,8),f(q.mape_percent,4),f(q.cost1,4),f(q.cost2,4),f(q.cost_total,4)] for q in a.itertuples()],size=8.2,widths=[.8,.5,1.25,1,1.1,1.1,1.2])
    r.p('本附录显示值为便于阅读而四舍五入；未四舍五入的完整结果在随报告附带的evidence/experiment1_results.csv。14节点逐值结果在experiment1_predictions_14_buses.csv；完整求解费用及变量在原研究目录cost_traces.json。')
    r.p('本报告全部数值表由已提交CSV/JSON自动读取，source_manifest.json记录来源文件校验值；未填充缺失实验或虚构实验结果。')
    r.save('实验报告一_单样本温度扰动')


def report_two(d):
    r=Report('02_cost_attribution','实验报告二：CNN 输出节点成本诊断与多日期验证','输出节点反事实、抽样Shapley与分层四模型经济性对照')
    c=d['cohort'];v=d['v2'];s=d['s2'];b=d['b'];ranks=s['reference_rankings']
    r.page('实验报告二｜节点成本诊断与多日期验证')
    r.p('从单点影响路径扩展到输出节点诊断和多日期证据',bold=True)
    r.p('整理日期：'+STAMP+'。本报告基于既有冻结CNN与四种预测头的实际评估记录，不重新训练模型、不改变Cost目标或数学约束。')
    r.table(['记录项','说明'],[
        ['结果版本','analysis/cnn-cost-attribution；提交 b527981'],
        ['承接实验','复用实验一完全相同的四种冻结模型、训练σ和五档扰动'],
        ['日期与负荷','27个预先分层的新日期 + 原参考点，合计28个不同日期'],
        ['正式规模','560个模型场景、7840个预测值、8680次Cost评估、17360个QP'],
        ['节点诊断','隔离变化、从完整输出复位单节点；参考点128条抽样Shapley顺序'],
        ['测试范围','2019-01-01至03-14，仅seed42、输入节点1温度']],widths=[1,3.5])
    r.h('摘要')
    r.p('实验验证了最大预测变化输出节点与最大Cost影响节点是否相同，并比较TAMU在不同日期、不同真实负荷水平下的经济性。原始CNN的112组非零扰动对照中，预测峰值与隔离/复位成本峰值均一致；参考点的抽样Shapley检查也一致。隔离影响之和与联合Cost变化最大相差3.246397，说明不能将逐节点隔离影响直接当作严格加性贡献。')
    r.p('27个新增日期的108个非零扰动场景中，TAMU有106个比直接遗忘更便宜；五档分层平均Cost均较原始/直接遗忘低。但在3月12日的−1σ反例中，TAMU反而比直接遗忘贵60.929210，平均优势不构成逐场景保证。')
    r.p('正式实验全部QP返回optimal，最大约束残差7.38e−8；当时累计57项测试通过，完整复算一致。多日期结果仍只覆盖现有前73天测试期，不能推出全年、夏季、其他种子或ERCOT上的普遍结论。')

    r.page('1  研究问题与分层实验设计')
    r.p('问题一：预测变化最大的输出节点是否也是Cost影响幅度最大的节点？问题二：在相同温度扰动和真实负荷下，直接遗忘、重拟合头与TAMU的绝对经济性和响应是否改变？问题三：单点发现能否在预先选定的其他日期中观察到？')
    r.table(['固定项','设置'],[
        ['模型与温度','实验一四种冻结预测头；input[0,0,0,4]；σ=0.935414141938编码单位'],
        ['扰动','−1、−0.5、0、+0.5、+1训练σ；仅改变一个输入坐标'],
        ['控制','每个样本内部固定真实负荷、其他139输入、权重和系统约束'],
        ['月份分层','1月、2月、3月；3月仅现有测试期的前14天'],
        ['月内负荷分层','按真实总负荷的1/3、2/3分位分为低/中/高'],
        ['每层取样','最接近层内10%、50%、90%负荷分位的小时，并约束日期不重复'],
        ['样本数','3个月 × 3负荷层 × 3日期 = 27；另保留参考样本0，共28日期'],
        ['汇总分母','分层平均与胜出率只用27个新日期，参考点不额外加权']],size=9.4,widths=[1,3.7])
    r.p('选样只使用日期和真实总负荷，不查看任何模型预测或Cost。选择清单在推理前冻结；1752个测试时刻的真实总负荷选样池一并保存，能够独立重算选择规则。低/中/高仅是各月内部的相对水平。')
    r.p('数据仍来自官方TX-123BT合成测试系统，真实负荷指数据中的ground truth，而非额外接入的现实电网观测。输入、输出缩放、两阶段费用与约束均继承实验一。')

    r.page('2  输出节点成本诊断方法')
    r.p('记零扰动预测为f0、完整温度扰动后的预测为f1，C(f)表示在固定真实负荷下，按原两阶段优化求得的总Cost。全部14个坐标是预测输出节点，不是14个温度输入实验。')
    r.table(['方法','定义与解释'],[
        ['隔离变化','仅令节点j采用f1[j]，其他输出保持f0；计算C(仅j变化)−C(f0)'],
        ['复位诊断','以完整f1为背景，仅将j复位为f0[j]；计算C(f1)−C(仅j复位)'],
        ['抽样Shapley','按不同顺序逐一加入输出变化，平均每个节点对C的边际贡献']],size=10,widths=[1,3.5])
    r.p('隔离与复位诊断对28个样本、4个非零扰动、14个输出节点执行，生成1568行诊断记录，每行包含两种影响。零扰动的Cost已在主实验实算，但没有变化的节点无需重复做同值诊断。')
    r.h('参考样本的抽样Shapley')
    r.p('仅在原始CNN参考样本0上，对四个非零温度水平进行Shapley分摊。使用64个独立随机排列及其逆序，共128条路径，seed42。相同输出子集的成本缓存复用，但每个新增子集都实际求解原两阶段QP。')
    r.code('贡献_j(π) = C(π中j之前的集合 ∪ {j}) − C(π中j之前的集合)\nShapley_j = 64个逆序对边际均值的平均\nMC标准误 = 对均值的样本标准差 / sqrt(64)')
    r.p('每条路径的边际值望远镜求和等于C(f1)−C(f0)。守恒仅验证分摊路径一致性，不意味着精确枚举了全部2^14个子集。MC标准误只反映顺序抽样误差，不包含求解误差、日期样本不确定性或模型训练不确定性。')
    r.p('混合输出向量未必对应任何可实现的真实CNN输入。这些方法是模型输出的反事实成本诊断，不能直接解释为物理控制因果，也不能把隔离影响之和当成严格的总成本分解。',bold=True)

    r.page('3  节点诊断的实际结果')
    r.figure('r2_node_attribution','图1  参考样本+1σ：预测变化与抽样Shapley成本分摊')
    r.table(['扰动/σ','预测峰值节点','隔离Cost峰值','复位Cost峰值','Shapley峰值'],[
        [f'{q["sigma_level"]:+g}',q['largest_forecast_change_bus'],q['largest_isolated_cost_bus'],q['largest_reset_cost_bus'],q['shapley_largest_bus']] for q in ranks],size=9.5)
    r.p('在本参考样本中，−1σ、−0.5σ、+0.5σ的峰值为节点3，+1σ的峰值为节点4。+1σ下节点4预测变化−0.278248 MW，抽样Shapley分配ΔCost约−2.289145。这里按绝对幅度排名，符号仍保留。')
    r.p('跨28个日期的112组非零扰动，预测峰值与隔离/复位峰值均一致（不一致0次）。两种诊断的前两名绝对差均大于1e−4诊断阈值，但该阈值不是目标值误差的严格上界。不能据本批一致，推出“预测变化最大必然等于成本影响最大”。')
    r.p('隔离影响之和与联合ΔCost的最大差为3.246397。参考点Shapley总分配与联合ΔCost最大误差3.55e−15；其MC标准误约1e−9，小于求解误差量级时不应当作有效小数位或总体置信区间。')

    r.page('4  多日期四模型经济性比较')
    r.figure('r2_cohort_mean_cost','图2  27个新日期的平均Cost及TAMU−直接遗忘配对差')
    r.table(['σ','原始均Cost','直接遗忘','重拟合头','TAMU','低于直接','低于原始'],[
        [f'{q.sigma_level:+g}',f(q.original_mean_cost,3),f(q.direct_mean_cost,3),f(q.retrained_mean_cost,3),f(q.tamu_mean_cost,3),f'{int(q.tamu_vs_direct_wins)}/27',f'{int(q.tamu_vs_original_wins)}/27'] for q in c.itertuples()],size=8.5,widths=[.5,1,1,1,1,.9,.9])
    r.p('胜出以配对差值<−1e−4判定。所有模型使用同一批输入、同一真实负荷、相同Cost定义。均值在五档上均显示TAMU较原始和直接遗忘更便宜；108个非零场景中，TAMU相对直接遗忘胜出106个，相对原始胜出85个。')
    r.p('阴影为所选日期间的25%—75%分位区间，不是总体置信区间。27个日期不是随机总体抽样，同一模型下不同日期也可能相关；这些结果是当前固定模型的描述性对照。')

    r.page('5  月份与真实负荷分层结果')
    r.figure('r2_month_load_strata','图3  每层3个不同日期；负值表示TAMU平均成本低于直接遗忘',width=6.35)
    r.p('该图对1/2/3月的低、中、高负荷层分别汇总，避免只展示总体平均值。3月低负荷层的优势较弱，不能以其他层的较大改善掩盖这一差异。这里的低/中/高仅按对应月份内的真实总负荷划分，不代表全年绝对负荷等级。')
    r.p('日期清单和负荷阈值保存在selection.json与parent_provenance.json；选样池保存在cohort_inputs.npz。编制报告时没有增删日期，没有根据本轮Cost表现替换样本。')
    r.p('参考样本0不计入分层平均，防止在1月额外加权。绘图和均值均从paired_model_costs.csv与cohort_summary.csv读取，可与原始逐场景results.csv核对。')

    r.page('6  必须保留的反例')
    r.figure('r2_counterexample','图4  2019-03-12，源Hour 7，测试索引1686，月内低负荷层')
    worst=b[(b.test_index==1686)&(b.sigma_level==-1.)]
    r.table(['模型','cost1','cost2','总Cost'],[[MODEL_NAMES[q.model],f(q.cost1),f(q.cost2),f(q.cost_total)] for q in worst.itertuples()],size=9.3,widths=[1.5,1,1,1])
    r.p('在同一真实总负荷478.328583 MW下，−1σ时TAMU总Cost为1209.299538，直接遗忘为1148.370328，前者更贵60.929210。该日期−0.5σ和零扰动也存在TAMU高于直接遗忘的情况；正向扰动后两者关系发生变化。')
    r.p('这一反例说明：整体平均Cost改善不等于每个日期、每个方向都更经济。TAMU使用原测试平均Cost分数恢复权重，没有对本次温度干预场景重新优化，也没有逐场景经济性约束。')

    r.page('7  工程验收与可复现性')
    r.table(['验收项','实际记录'],[
        ['固定模型','四种快照与实验一字节相同；干预前后模型哈希不变'],
        ['样本控制','28个不同日期；每次固定对应真实负荷；零扰动预测逐位一致、Cost差0'],
        ['正式评估','560个模型场景、7840个预测；8680次Cost评估，即17360个QP'],
        ['节点/子集','1568行节点诊断；8232条子集成本记录（含复用主场景端点）'],
        ['求解状态','全部optimal，实际使用GUROBI'],
        ['约束与费用','最大约束残差7.38e−8；七项费用分解最大误差4.55e−13'],
        ['累计回归测试','57项通过，0失败、0跳过；已包含实验一的49项，不能相加为106项'],
        ['独立重放','7840个预测逐位一致；全部成本CSV与求解轨迹逐字节一致'],
        ['下载包','在无Git元数据、无年度数据的解压目录完成57项测试及完整重放']],size=9.2,widths=[1,3.6])
    r.h('复算入口')
    r.code('.venv/bin/python experiments/cost_attribution.py \\\n  --replay-from research/cnn_cost_attribution \\\n  --output-dir /tmp/cost-path-report-replay')
    r.code('.venv/bin/python experiments/validate_cost_attribution.py \\\n  --replay-dir /tmp/cost-path-report-replay')
    r.p('本次报告制作只读取以上已保存记录，未重新执行实验。逐字节复算一致是原固定环境与相同求解器的验证；换平台或兼容求解器可能出现数值差异。')

    r.page('8  结论、局限与后续工作')
    r.h('本轮结论')
    r.bullet('已建立输入温度→多个输出预测→两阶段Cost的可追踪路径，并区分隔离、复位与Shapley诊断含义。')
    r.bullet('本批112组原始CNN对照的峰值节点一致，但逐节点隔离影响不是可直接相加的总成本贡献。')
    r.bullet('所选27个日期中TAMU平均经济性较好，但存在明确较贵反例；不能将平均优势升级为逐场景或因果保证。')
    r.h('局限')
    r.p('测试期只覆盖2019年前73天，没有夏季/全年验证；只有seed42、一个输入节点和温度一个特征。分层样本不是随机总体样本，模型训练和日期间不确定性未被本轮MC标准误刻画。混合输出向量也未必是CNN可实现输入。')
    r.p('原全年归一化先于划分，helpful/TAMU使用测试性能信息的协议限制仍然存在。本实验中的ground truth属于官方合成测试数据，不代表已经接入ERCOT现实观测。')
    r.h('建议下一步（尚未执行）')
    r.bullet('在相同固定模型上扩展其他物理输入特征，先明确各自训练尺度和编码合法性，避免任意破坏sin/cos关系。')
    r.bullet('用多个种子与预先确定的场景评估稳定性；把平均Cost、最差Cost和输入敏感性分别报告。')
    r.bullet('如需夏季/全年独立测试，应重新设计时间划分和归一化评估，不能把原训练段直接称为独立测试。')
    r.bullet('ERCOT正式Cost实验需先确定节点映射和负荷缩放方案；本轮没有开展该项工作。')
    r.table(['源码/数据目录','说明'],[
        ['experiments/cost_attribution.py','冻结推理、反事实、抽样路径及实际Cost保存'],
        ['experiments/validate_cost_attribution.py','样本池重算、全部真实负荷/预测向量、子集费用、重放验收'],
        ['research/cnn_cost_attribution/','原始结果、完整8680次Cost轨迹及模型/输入快照'],
        ['本报告evidence/与source_manifest.json','用于文档表格的原数据副本与来源SHA256']],size=9,widths=[1.7,2.6])

    r.page('附录A  参考点+1σ的逐节点分摊')
    sh=d['shap'][d['shap'].sigma_level==1].sort_values('grid_bus')
    r.table(['输出节点','Δ预测/MW','Shapley cost1','Shapley cost2','Shapley总计'],[
        [int(q.grid_bus),signed(q.prediction_delta_mw),signed(q.shapley_cost1),signed(q.shapley_cost2),signed(q.shapley_total)] for q in sh.itertuples()],size=9)
    r.p('表中Shapley是128条顺序的抽样估计，只适用于原始CNN参考点+1σ，不是对全部28日期的精确Shapley枚举。MC误差单独保存在reference_shapley.csv，不能当作求解精度或总体置信区间。')
    r.h('汇总分母说明')
    r.table(['统计','分母'],[
        ['原始CNN节点峰值比较','28个日期 × 4个非零温度水平 = 112组'],
        ['TAMU非零经济性比较','27个新日期 × 4个非零水平 = 108场景；106场景低于直接遗忘'],
        ['四模型主场景','28个日期 × 4模型 × 5温度水平 = 560组'],
        ['分层平均成本','每个温度水平27个新日期；原参考点不计入'],
        ['回归测试','实验二时累计57项；不与实验一49项相加']],size=9.4,widths=[1.5,3])
    r.save('实验报告二_节点成本诊断与多日期验证')


def textbox(slide,x,y,w,h,text,size=20,color=NAVY,bold=False):
    shape=slide.shapes.add_textbox(PInches(x),PInches(y),PInches(w),PInches(h))
    tf=shape.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=PInches(.02)
    tf.margin_top=tf.margin_bottom=PInches(.02)
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.space_after=PPt(8);p.font.name=FONT;p.font.size=PPt(size)
        p.font.color.rgb=PColor.from_string(color);p.font.bold=bold;p.text=line
    return shape


def rect(slide,x,y,w,h,color=LIGHT):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,PInches(x),PInches(y),PInches(w),PInches(h))
    shape.fill.solid();shape.fill.fore_color.rgb=PColor.from_string(color);shape.line.fill.background()
    return shape


def fit_image(slide,name,x,y,w,h):
    from PIL import Image
    path=ASSETS/f'{name}.png'
    with Image.open(path) as im:ratio=im.width/im.height
    width=min(w,h*ratio);height=width/ratio
    slide.shapes.add_picture(str(path),PInches(x+(w-width)/2),PInches(y+(h-height)/2),width=PInches(width),height=PInches(height))


def ppt_table(slide,headers,rows,x,y,w,h,widths=None,size=17):
    shape=slide.shapes.add_table(len(rows)+1,len(headers),PInches(x),PInches(y),PInches(w),PInches(h))
    table=shape.table
    if widths:
        for col,width in zip(table.columns,widths):col.width=PInches(w*width/sum(widths))
    for i,row in enumerate([headers,*rows]):
        for j,text in enumerate(row):
            cell=table.cell(i,j);cell.text=str(text)
            cell.fill.solid();cell.fill.fore_color.rgb=PColor.from_string(NAVY if i==0 else ('F0F5F6' if i%2 else 'FFFFFF'))
            cell.margin_left=cell.margin_right=PInches(.08);cell.margin_top=cell.margin_bottom=PInches(.04)
            for p in cell.text_frame.paragraphs:
                p.font.name=FONT;p.font.size=PPt(size);p.font.bold=i==0
                p.font.color.rgb=PColor.from_string('FFFFFF' if i==0 else NAVY)
    return shape


def build_deck(d):
    folder=OUT/'03_group_meeting';folder.mkdir(exist_ok=True)
    prs=Presentation();prs.slide_width=PInches(13.333);prs.slide_height=PInches(7.5)
    prs.core_properties.title='CNN输入扰动对预测与运行成本的影响：两轮实验组会汇报'
    notes=[]
    def new(title,subtitle,note):
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid();slide.background.fill.fore_color.rgb=PColor.from_string('FFFFFF')
        bar=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,PInches(.12));bar.fill.solid();bar.fill.fore_color.rgb=PColor.from_string(TEAL);bar.line.fill.background()
        textbox(slide,.42,.32,12.4,.55,title,size=29,bold=True)
        textbox(slide,.45,.99,12.4,.47,subtitle,size=15,color=GRAY)
        textbox(slide,.46,7.15,11.8,.22,'TX-123BT / case14  |  固定CNN与原两阶段Cost  |  整理 '+STAMP,size=9,color=GRAY)
        textbox(slide,12.35,7.12,.55,.3,f'{len(prs.slides):02d}',size=12,color=TEAL)
        slide.notes_slide.notes_text_frame.text=note
        notes.append((title,subtitle,note));return slide
    s=new('CNN 输入扰动如何影响负荷预测与运行成本','两轮实验进展｜组会汇报｜目标汇报时长约12–15分钟',
        '本次汇报围绕导师提出的要求：改变十个输入特征中的一个，观察预测输出，并找到它到下游运行成本的影响路径。现阶段复用已经修复的CNN和两阶段优化，不重新训练。汇报分为单样本路径验证、节点成本诊断、多日期经济性三部分。')
    textbox(s,.6,1.85,11.9,1.2,'从一个温度坐标出发，追踪到\n14个节点预测与两阶段电网运行成本',size=31,bold=True)
    for x,title,sub in [(.65,'实验一','单样本 · 五档扰动 · 四模型'),(4.85,'实验二','节点诊断 · 28日期 · 分层对照'),(9.05,'保留边界','不把数值干预当作物理因果')]:
        rect(s,x,4.15,3.65,1.55);textbox(s,x+.18,4.35,3.3,.45,title,22,bold=True,color=TEAL);textbox(s,x+.18,4.93,3.25,.6,sub,16)
    s=new('本阶段得到三条可展示的结论','先报告观察到的事实，再说明它们不支持哪些推断',
        '第一条结论是路径已经打通：温度编码经过固定CNN影响多个输出，再经调度和再调度影响Cost。第二条结论是，本批112组原始CNN比较中，预测峰值与两类成本诊断峰值一致，不能为了预设叙事强行说二者不同。第三条是TAMU在所选分层日期中平均成本较低，但有真实反例，不能当成每个场景的保证。')
    for i,(a,b) in enumerate([('路径已打通','输入温度 → 14节点预测 → 调度 / 再调度 → Cost'),('峰值在本批一致','112组原始CNN非零扰动对照，隔离 / 复位诊断均一致'),('TAMU有平均优势，也有例外','27个新日期的108个非零场景中，106个低于直接遗忘')]):
        rect(s,.6,1.75+i*1.57,12.05,1.26);textbox(s,.85,1.91+i*1.57,3.0,.47,a,22,bold=True,color=TEAL);textbox(s,4.0,1.96+i*1.57,8.3,.72,b,20)
    s=new('固定模型与实验边界','保留原论文Cost定义、目标函数、数学约束及数据语义',
        '数据是官方TX-123BT的2019合成测试系统数据。模型为CNN，种子42，helpful删除20%的敏感样本。四种对照共享同一个冻结卷积核心，所谓重训练仅指原项目协议下910个仿射参数的重新拟合。TAMU采用平均L1等于0.1的既有恢复权重，本轮不针对温度扰动重新优化。当前测试集是全年前1752小时，也就是前73天。')
    ppt_table(s,['项目','固定设置'],[
        ['数据 / 电网','官方 TX-123BT 2019 合成数据 / case14'],['模型与删除','CNN；seed42；helpful；20%敏感样本删除'],
        ['四种对照','原始 / 直接遗忘 / 剩余样本重拟合头 / TAMU'],['TAMU与模型范围','平均L1=0.1；卷积核心固定，只比较预测头'],
        ['测试覆盖','1752小时：2019-01-01 至 03-14；没有夏季验证']],.65,1.75,12,4.5,[1.1,3.8],size=20)
    s=new('输入有140个编码值，64维是隐藏表示','温度：节点内零基索引4；本次位置 input[0,0,0,4]',
        '需要先纠正输入的概念。每个时刻有14个节点，每节点10个特征，所以原始编码输入是140维。经过两层卷积、展平和线性层后，得到64维tanh隐藏表示，不能给每个隐藏维度直接取一个物理特征名。本次只改输入节点1的温度，也就是张量的零基位置0、0、0、4。')
    for x,title,sub in [(.6,'输入','1 × 1 × 14 × 10'),(4.75,'冻结 CNN','卷积 → 280 → tanh 64'),(8.9,'预测输出','64 → 14节点')]:
        rect(s,x,2.0,3.8,1.3);textbox(s,x+.16,2.16,3.4,.4,title,23,bold=True,color=TEAL);textbox(s,x+.16,2.73,3.42,.42,sub,19)
    textbox(s,.7,3.8,11.9,1.55,'每节点10个特征：\n星期sin/cos、小时sin/cos、温度、短波/长波辐射、纬向/经向风速、风速',size=22)
    textbox(s,.7,5.7,11.8,.7,'目标Load不属于输入；预测先反标准化，再以p.u.进入Cost程序。',size=22,bold=True,color=TEAL)
    s=new('实验一：只改变一个温度坐标','原始样本 test index 0；其余139个输入、权重与真实负荷固定',
        '温度扰动采用加法，而不是把标准化值乘以百分比。训练集7008行给出的温度标准差约为0.935414个编码单位，对应7.936348K的温差。原模型全年归一化参数保持不变。五档输入共享同一条真实负荷，目的是测模型响应，不是模拟实际发生的天气。零扰动必须逐位恢复预测与Cost。')
    rect(s,.65,1.8,12,1.25);textbox(s,.95,2.05,11.3,.6,'x温度(k) = x温度(0) + k × σ训练',size=31,bold=True)
    for i,t in enumerate(['−1σ','−0.5σ','0','+0.5σ','+1σ']):
        rect(s,.8+i*2.49,3.55,2.15,.95,'DCEEEF' if i!=2 else 'D8E8F4');textbox(s,.92+i*2.49,3.78,1.9,.5,t,27,bold=True,color=TEAL)
    textbox(s,.85,5.1,11.5,1.05,'σ训练 = 0.935414141938 编码单位 ≈ 7.936348491 K\n只取训练段统计；零扰动作为独立验收控制。',size=23)
    s=new('改变一个温度输入，多个输出节点都发生变化','实验一，原始CNN，参考样本 +1σ；图中为相对零扰动的变化',
        '对参考样本加一个训练标准差的温度扰动，原始CNN的14个输出节点都发生了大于一百万分之一MW的变化。最大绝对变化在输出节点4，为负0.278248MW，总预测负荷变化为负0.827542MW。这里输入节点是1，最大变化输出节点是4，两者不能混淆。完整五档曲线和逐节点表见实验报告一。')
    fit_image(s,'r1_plus1_node_changes',.55,1.65,8.55,4.95)
    rect(s,9.35,2,3.35,3.9);textbox(s,9.6,2.25,2.85,.65,'节点4最大',25,bold=True,color=TEAL)
    textbox(s,9.6,3.15,2.85,2.3,'−0.278248 MW\n\n总预测变化\n−0.827542 MW',size=23)
    s=new('预测变化如何进入电网运行成本','预测负荷用于阶段一；阶段二仍使用该样本固定的真实负荷',
        '运行成本没有跳过计算。阶段一根据预测负荷调度发电，产生cost1；阶段二拿阶段一的PG和固定真实负荷再调度，产生cost2。对于参考样本正一个标准差，cost1下降0.481661，cost2下降6.296702，总Cost下降6.778363。这次主要是阶段二过量发电处理费用变化，不应推广成所有样本的机制。')
    fit_image(s,'r1_two_stage_cost',.45,1.65,9.0,4.8)
    rect(s,9.65,2.05,3.0,3.75);textbox(s,9.87,2.28,2.55,3.2,'+1σ\nΔcost1 −0.481661\nΔcost2 −6.296702\n\nΔ总Cost −6.778363',size=21,bold=True)
    s=new('绝对经济性和响应幅度，是两个不同问题','实验一：四种预测头、相同五档输入、相同真实负荷',
        '左图比较绝对总Cost，右图比较每个模型相对自己零扰动的变化。直接遗忘和重拟合预测头几乎重叠。TAMU在本参考点的绝对成本较低，正向扰动的变化幅度也较小，但负向扰动的变化幅度更大。因此应分别回答是否经济、是否敏感，而不是用一个指标代替另一个。')
    fit_image(s,'r1_four_model_cost',.45,1.62,12.4,4.85)
    textbox(s,.8,6.55,11.9,.4,'TAMU：+1σ的 |ΔCost| 较小；−1σ的 |ΔCost| 反而较大。',size=20,bold=True,color=TEAL)
    s=new('不能把预测变化量直接叫作成本贡献','节点诊断只干预预测输出向量；每个混合向量都重解原两阶段Cost',
        '为回答预测变化最大节点是否就是Cost影响最大节点，设计了两种对照。隔离诊断以零扰动为背景，只保留一个输出变化；复位诊断以完整扰动为背景，只复位一个输出。参考样本还用128条顺序的抽样Shapley，考虑不同加入顺序下的边际成本。混合输出不一定来自可实现的CNN输入，因此这是输出反事实诊断，不是物理控制因果。')
    for y,title,body in [(1.75,'隔离变化','只有输出节点j采用扰动后的预测，其余保持零扰动'),(3.25,'复位诊断','从完整扰动向量中，只把输出节点j复位到零扰动'),(4.75,'抽样 Shapley','64个独立排列及其逆序：128条路径，平均边际成本')]:
        rect(s,.65,y,12,1.15);textbox(s,.9,y+.2,2.2,.6,title,23,bold=True,color=TEAL);textbox(s,3.2,y+.22,9.0,.65,body,20)
    s=new('本批峰值节点一致，但隔离影响不可直接相加','参考点 +1σ：输出节点4的抽样Shapley约为 −2.289145',
        '在原始CNN的28个日期、四档非零扰动，也就是112组比较中，预测峰值与隔离、复位Cost峰值都是一致的。参考样本Shapley检查也一致。这个结果不能被写成必然规律。另外，隔离影响相加与联合Cost变化最大相差3.246397，说明不能把逐节点隔离值直接堆成一个严格的成本贡献图。Shapley守恒是路径一致性，不是物理因果。')
    fit_image(s,'r2_node_attribution',.4,1.65,12.55,4.6)
    textbox(s,.75,6.38,12,.55,'112组峰值一致 ≠ 普遍规律；隔离求和与联合 ΔCost 最大差 3.246397。',size=20,bold=True,color=TEAL)
    s=new('实验二：预先选择不同日期和真实负荷水平','选样不看模型预测或Cost；原参考点不计入27个新日期的平均值',
        '为减少只看一个时刻的偶然性，按测试期的1月、2月、3月分别划分真实总负荷的低中高三层。每层取三个互不重复日期，目标是接近层内10%、50%、90%的负荷分位。得到27个新日期，加原参考点为28个日期。选样池和选择清单先保存，再推理。当前3月只覆盖前14天，所以不能说已经做了全年验证。')
    ppt_table(s,['月内真实负荷层','1月','2月','3月（仅前14天）'],[['低','3个日期','3个日期','3个日期'],['中','3个日期','3个日期','3个日期'],['高','3个日期','3个日期','3个日期']],.7,1.85,11.95,2.7,[1.4,1,1,1.5],size=23)
    textbox(s,.85,5.02,11.8,1.22,'27 个新日期 + 1 个参考日期 = 28 个不同日期\n28 × 4模型 × 5温度水平 = 560 组主场景',size=27,bold=True,color=TEAL)
    s=new('分层平均Cost较低，但不等于每个场景更优','27个新日期；配对比较中的真实负荷、输入和优化约束相同',
        '五档扰动下，TAMU在所选27个日期上的平均Cost都低于原始模型和直接遗忘。图中右侧是TAMU减直接遗忘的配对成本差，阴影是日期间四分位区间，不是总体置信区间。排除零扰动后有108个场景，其中106个TAMU更便宜。要保留另外两个非零反例，并避免用平均优势宣称逐样本保证。')
    fit_image(s,'r2_cohort_mean_cost',.4,1.6,12.55,4.9)
    textbox(s,.8,6.55,11.9,.4,'108 个非零场景中，106 个 TAMU 低于直接遗忘；阴影不是总体置信区间。',size=19,bold=True,color=TEAL)
    s=new('反例：3月12日 −1σ，TAMU反而更贵','test index 1686，源Hour 7，月内低负荷；固定真实总负荷478.328583 MW',
        '必须主动报告这个反例。同一条真实负荷、负一个标准差时，TAMU的Cost为1209.299538，直接遗忘为1148.370328，TAMU更贵60.929210。正向扰动后它又低于直接遗忘。这里不能通过删掉这个日期来让结果好看；该日期来自事前分层规则，已经保留在所有汇总里。')
    fit_image(s,'r2_counterexample',.4,1.7,9.1,4.85)
    rect(s,9.75,2.1,2.85,3.9,'F7E9E9');textbox(s,9.98,2.35,2.4,3.1,'−1σ\nTAMU 1209.299538\n直接 1148.370328\n\n差值 +60.929210',size=20,bold=True,color=RED)
    s=new('验证充分，不代表推广范围无限','先区分工程正确性、描述性结果与泛化或因果结论',
        '工程方面，实验一当时49项测试通过，实验二扩展为累计57项，后者包含前者，不能相加。第二轮17360个QP全部optimal，完整复算的7840个预测逐位一致，成本CSV和轨迹逐字节一致。解释上仍有范围限制：一个种子、一个输入温度特征、前73天测试期；原项目全年归一化和测试分数选样的限制也未消除。Shapley抽样误差不能替代日期或模型的不确定性。')
    for x,y,title,text in [(.65,1.8,'工程验收','49 → 57项累计测试\n17,360个QP均optimal\n7,840个预测完整复算一致'),(7,1.8,'结果边界','seed42；输入节点1温度\n仅2019年前73天\n分层日期不是随机总体样本'),(.65,4.25,'原协议限制','全年归一化先于划分\nhelpful/TAMU使用测试分数\n尚未消除信息使用偏差'),(7,4.25,'不要混淆','平均经济性 ≠ 逐场景保证\nMC误差 ≠ 总体置信区间\n输出反事实 ≠ 物理因果')]:
        rect(s,x,y,5.65,2.15);textbox(s,x+.2,y+.17,5.2,.42,title,23,bold=True,color=TEAL);textbox(s,x+.2,y+.72,5.2,1.25,text,18)
    s=new('下一步：扩展证据，而不是扩大表述','本阶段：路径可追踪、对照可复算、反例已保留',
        '下一步建议先扩展其他物理特征，同时明确尺度和编码是否合法；再进行多个种子和预先确定场景的稳定性分析。若要讨论夏季或全年，必须重新设计独立时间划分和归一化评估，不能把原训练段直接叫作测试。ERCOT正式Cost实验仍需先确定节点映射。今天的核心交付是两份实验报告、图表、数据和可重复运行的脚本，而不是一个超出证据的普遍结论。')
    textbox(s,.8,1.9,11.7,1.0,'已经完成：输入影响路径 + 四模型对照 + 节点诊断 + 多日期验证',size=25,bold=True,color=TEAL)
    for i,line in enumerate(['扩展其他物理特征：明确训练尺度与编码合法性','扩展多个种子与预设场景：分别报告均值、最差值和敏感性','全年 / ERCOT：先设计独立测试与节点映射，再做正式Cost实验']):
        rect(s,.75,3.15+i*1.02,11.9,.82);textbox(s,.98,3.33+i*1.02,11.4,.5,f'{i+1}.  {line}',size=21)
    textbox(s,.85,6.55,11.7,.38,'完整数值、复现命令与局限：随附两份实验报告及 source_manifest.json。',size=17,color=GRAY)
    prs.save(folder/'组会汇报_CNN输入影响路径.pptx')
    build_notes(notes)
    return notes


def build_notes(notes):
    r=Report('03_group_meeting','组会汇报讲稿：CNN输入扰动的预测与成本影响','对应15页PPT；建议时长约12–15分钟；可按组会时间删减')
    for i,(title,subtitle,note) in enumerate(notes):
        if i%3==0:r.page(f'讲稿｜第{i+1}—{min(i+3,len(notes))}页')
        r.h(f'{i+1:02d}  {title}')
        r.p(note)
    r.page('答疑备忘与资料索引')
    for q,a in [
        ('为什么不是64个原始输入？','原始输入是14×10；64是经过卷积和线性层学习得到的隐藏表示，不能直接用物理特征名解释。'),
        ('+1σ为什么预测反而降低？','这是固定模型在固定样本上的局部数值响应，其他气象特征与真实负荷都不变，不能推断现实气温上升必然降低负荷。'),
        ('为什么TAMU平均更好却有反例？','原权重由测试平均Cost的一阶分数恢复，没有每个温度场景的成本约束或稳健性保证。'),
        ('预测峰值和成本峰值到底是否不同？','在本批112组原始CNN比较中，两类反事实诊断的峰值与预测峰值一致；不能预设必须不同，也不能据此证明普遍相等。'),
        ('Shapley误差很小能否说明结论稳定？','不能。这里MC标准误只衡量排列抽样，未包括求解误差、日期抽样、训练种子或其他模型的不确定性。'),
        ('“重训练CNN”是否重训了卷积层？','没有。沿用原项目冻结core、在剩余敏感样本重拟合910个仿射参数的协议。'),
        ('两个测试通过数能否相加？','不能。实验二57项是扩展后的累计测试集合，包含实验一49项。')]:
        r.h(q);r.p(a)
    r.p('实验一记录提交b5fd9f1；实验二记录提交b527981。所有展示数值来自随报告附带的evidence/，原项目完整模型快照和求解轨迹仍保存在research/对应目录。')
    r.save('组会汇报_讲稿')


def copy_evidence():
    folder=OUT/'evidence';folder.mkdir(exist_ok=True)
    sources={
        'experiment1_results.csv':E1/'results.csv',
        'experiment1_predictions_14_buses.csv':E1/'predictions_14_buses.csv',
        'experiment1_validation.json':E1/'validation.json',
        'experiment1_summary.json':E1/'summary.json',
        'experiment1_cost_traces.json':E1/'cost_traces.json',
        'experiment2_results.csv':E2/'results.csv',
        'experiment2_cohort_summary.csv':E2/'cohort_summary.csv',
        'experiment2_paired_model_costs.csv':E2/'paired_model_costs.csv',
        'experiment2_reference_shapley.csv':E2/'reference_shapley.csv',
        'experiment2_node_rank_comparison.csv':E2/'node_rank_comparison.csv',
        'experiment2_selection.json':E2/'selection.json',
        'experiment2_validation.json':E2/'validation.json',
        'experiment2_checks.json':E2/'checks.json',
        'experiment2_counterexamples.csv':E2/'economic_counterexamples.csv'}
    mapping=[]
    for name,path in sources.items():
        shutil.copyfile(path,folder/name)
        mapping.append(dict(copy='evidence/'+name,source=str(path.relative_to(ROOT)),sha256=sha(path)))
    return mapping


def main():
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(exist_ok=True)
    d=load();mapping=copy_evidence()
    draw_charts(d);report_one(d);report_two(d);notes=build_deck(d)
    versions={p:importlib.metadata.version(p) for p in ['python-docx','python-pptx','matplotlib','pandas','numpy','Pillow','PyMuPDF']}
    manifest=dict(compiled_date_shanghai=STAMP,
        experiment1_commit='b5fd9f13296382a787f28cacbcb498dd9fff07bc',
        experiment2_commit='b527981a9b5f6c1820a442a44f59ed6a9d446356',
        purpose='Two separate experiment reports plus one group-meeting report with editable slides and speaker manuscript',
        new_experiments_run=False,model_training_performed=False,source_evidence=mapping,
        experiments=dict(single_sample=dict(scenarios=20,tests_passed_at_that_stage=49,qp_solves=48),
            cohort=dict(contexts=28,stratified_dates=27,scenarios=560,forecast_values=7840,qp_solves=17360,
                cumulative_tests_passed=57,nonzero_paired_cases=108,tamu_lower_than_direct=106)),
        slides=len(notes),document_tool_versions=versions,
        builder_sha256=sha(Path(__file__)),
        report_chart_note='Charts redrawn from the saved CSV/JSON for readable document and slide typography; no values or experiments changed')
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),slides=len(notes),evidence_files=len(mapping)),ensure_ascii=False))


if __name__=='__main__':main()

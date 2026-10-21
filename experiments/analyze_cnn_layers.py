"""Fixed-input CNN internal-layer analysis; no input perturbation or cost solve.

Captum is an external optional analysis dependency. Use the documented source
checkout via PYTHONPATH; this script never downloads or executes uploaded ZIPs.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cloud-setup/task-aware/matplotlib')

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.net import NN_CONV
from utils.funcs import ModelNNAffine
from captum.attr import LayerActivation, LayerGradientXActivation
import captum

MODELS=['original','unlearned20','retrained','tamu20']
STAGES=['conv1','relu1','conv2','relu2','flatten','lin1','tanh','head']
EXPECTED=[(1,8,14,10),(1,8,14,10),(1,8,7,5),(1,8,7,5),
          (1,280),(1,64),(1,64),(1,14)]
ROLES=['3×3邻域特征提取','负响应截零；与第二次ReLU分开记录',
       '4×4卷积、步长2；局部感受野扩大','第二次负响应截零',
       '只改排列：8×7×5→280','全连接融合：280→64',
       '限制到(-1,1)；可能出现饱和','仿射读出：64→14；四模型主要差异所在']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


class LayerView(nn.Module):
    """Equivalent predictor with unique modules for the two ReLU calls."""
    def __init__(self,core,head,mean,std):
        super().__init__()
        self.conv1=copy.deepcopy(core.conv1)
        self.relu1=nn.ReLU()
        self.conv2=copy.deepcopy(core.conv2)
        self.relu2=nn.ReLU()
        self.flatten=nn.Flatten(start_dim=1)
        self.lin1=copy.deepcopy(core.lin1)
        self.tanh=copy.deepcopy(core.tanh)
        self.head=copy.deepcopy(head.fc)
        self.register_buffer('mean',torch.from_numpy(mean).double())
        self.register_buffer('std',torch.from_numpy(std).double())

    def traced(self,x):
        activations={}
        for name in STAGES:
            if name=='head':x=x.to(self.head.weight)
            x=getattr(self,name)(x);activations[name]=x
        return (x*self.std+self.mean)*100.,activations

    def forward(self,x):return self.traced(x)[0]


def digest_model(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(args):
    torch.set_num_threads(1);torch.manual_seed(42)
    source=args.source.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    weights=source/'frozen_models.npz';inputs=source/'reference_inputs.npz'
    source_before={str(p):sha(p) for p in [weights,inputs]}
    snapshot=np.load(weights,allow_pickle=False);data=np.load(inputs,allow_pickle=False)
    raw=data['raw'].copy();mean=data['mean'].copy();std=data['std'].copy()
    core=NN_CONV(in_ch=1,input_length=14,input_width=10,width=2,linear_size=64,output_size=14).eval()
    core.load_state_dict({k:torch.from_numpy(snapshot['core.'+k]) for k in core.state_dict()})
    x=torch.from_numpy(raw).requires_grad_(True)
    activation_rows=[];attr_rows=[];ablation_rows=[];prediction_rows=[]
    archive={};checks=[];model_hashes={};manual_errors=[];activation_errors=[];head_errors=[];ablation_errors=[]
    original_acts=None;total_ablations=0

    def check(name,condition,**details):
        if not bool(condition):raise AssertionError((name,details))
        checks.append(dict(name=name,passed=True,**details))

    # Verify the original shared ReLU executes twice with different tensor shapes.
    relu_hits=[]
    hook=core.relu.register_forward_hook(lambda module,ins,outs:relu_hits.append(tuple(outs.shape)))
    with torch.no_grad():core(torch.from_numpy(raw))
    hook.remove();check('original_shared_relu_has_two_calls',relu_hits==[EXPECTED[1],EXPECTED[3]],shapes=relu_hits)

    for model_name in MODELS:
        parameter=np.column_stack([snapshot[model_name+'.weight'],snapshot[model_name+'.bias']]).reshape(-1)
        head=ModelNNAffine(parameter).eval()
        model=LayerView(core,head,mean,std).eval();before=digest_model(model)
        prediction,acts=model.traced(x)
        with torch.no_grad():
            latent=core(torch.from_numpy(raw))[0]
            reference=(head(latent)*torch.from_numpy(std)+torch.from_numpy(mean))*100.
        check(model_name+'_wrapper_forecast_bit_identical',np.array_equal(prediction.detach().numpy(),reference.numpy()))
        check(model_name+'_parameter_count',sum(p.numel() for p in model.parameters())==20006)
        if original_acts is None:original_acts={n:acts[n].detach().numpy().copy() for n in STAGES[:-1]}
        check(model_name+'_shared_core_activations_identical',all(np.array_equal(acts[n].detach().numpy(),original_acts[n]) for n in STAGES[:-1]))
        check(model_name+'_stage_shapes',[tuple(acts[n].shape) for n in STAGES]==EXPECTED)
        archive[model_name+'.prediction_mw']=prediction.detach().numpy()
        for target in range(14):
            prediction_rows.append(dict(model=model_name,output_bus=target+1,prediction_mw=float(prediction[0,target].detach())))
        for name,role in zip(STAGES,ROLES):
            a=acts[name].detach().numpy();archive[model_name+'.activation.'+name]=a
            measured=LayerActivation(model,getattr(model,name)).attribute(x)
            err=float(np.max(np.abs(a-measured.detach().numpy())));activation_errors.append(err)
            activation_rows.append(dict(model=model_name,layer=name,shape='x'.join(map(str,a.shape)),
                elements=int(a.size),parameters=sum(p.numel() for p in getattr(model,name).parameters()),
                minimum=float(a.min()),maximum=float(a.max()),mean=float(a.mean()),std=float(a.std()),
                zero_fraction=float(np.mean(a==0)),fraction_abs_above_095=float(np.mean(np.abs(a)>.95)) if name=='tanh' else '',role=role))
        for target in range(14):
            gradients=torch.autograd.grad(prediction[0,target],tuple(acts.values()),retain_graph=True)
            for name,gradient in zip(STAGES,gradients):
                analytic=(gradient*acts[name]).detach().numpy()
                attribution=LayerGradientXActivation(model,getattr(model,name)).attribute(x,target=target).detach().numpy()
                error=float(np.max(np.abs(analytic-attribution)));manual_errors.append(error)
                archive[f'{model_name}.gxa.{name}.output{target+1}']=attribution
                if attribution.ndim==4:
                    signed=attribution[0].sum(axis=(1,2));magnitude=np.abs(attribution[0]).sum(axis=(1,2))
                    kind='channel'
                else:
                    signed=attribution[0];magnitude=np.abs(signed);kind='unit'
                for i,(value,mag) in enumerate(zip(signed,magnitude)):
                    attr_rows.append(dict(model=model_name,layer=name,output_bus=target+1,
                        unit_index0=i,unit_label1=i+1,unit_kind=kind,
                        signed_gradient_x_activation_mw=float(value),absolute_mass_mw=float(mag)))
        # The final affine head admits an exact within-model decomposition.
        hidden=acts['tanh'][0].detach().numpy().astype(np.float64)
        factor=std.astype(np.float64)*100.
        contribution=head.fc.weight.detach().numpy()*hidden[None,:]*factor[:,None]
        intercept=(head.fc.bias.detach().numpy()*std.astype(np.float64)+mean.astype(np.float64))*100.
        archive[model_name+'.head_contribution_mw']=contribution
        archive[model_name+'.head_intercept_mw']=intercept
        error=float(np.max(np.abs(contribution.sum(1)+intercept-prediction[0].detach().numpy())))
        head_errors.append(error)
        check(model_name+'_affine_head_decomposition',error<1e-10,error=error)
        for target in range(14):
            gxa=archive[f'{model_name}.gxa.tanh.output{target+1}'][0]
            check(model_name+f'_hidden_gxa_equals_affine_contribution_output{target+1}',np.allclose(gxa,contribution[target],rtol=2e-7,atol=2e-5))
        # Internal ablations, not perturbations of any raw input coordinate.
        for name,count in [('relu1',8),('relu2',8),('tanh',64)]:
            for unit in range(count):
                def remove_unit(module,ins,output,index=unit):
                    result=output.clone();result[:,index]=0;return result
                hook=getattr(model,name).register_forward_hook(remove_unit)
                with torch.no_grad():ablated=model(torch.from_numpy(raw)).numpy()[0]
                hook.remove();delta=ablated-prediction[0].detach().numpy()
                total_ablations+=1
                for target in range(14):
                    ablation_rows.append(dict(model=model_name,layer=name,unit_index0=unit,unit_label1=unit+1,
                        unit_kind='channel' if name.startswith('relu') else 'hidden_neuron',output_bus=target+1,
                        original_prediction_mw=float(prediction[0,target].detach()),ablated_prediction_mw=float(ablated[target]),
                        delta_prediction_mw=float(delta[target])))
                if name=='tanh':ablation_errors.append(float(np.max(np.abs(delta+contribution[:,unit]))))
        check(model_name+'_parameters_unchanged',digest_model(model)==before)
        model_hashes[model_name]=before
    check('captum_activations_match_trace',max(activation_errors)==0,max_error=max(activation_errors))
    check('captum_gxa_matches_direct_autograd',max(manual_errors)<1e-8,max_error=max(manual_errors))
    check('all_outputs_finite',all(np.isfinite(a).all() for a in archive.values()))
    check('hidden_neuron_ablation_matches_exact_head',max(ablation_errors)<1e-10,max_error=max(ablation_errors))
    check('source_files_unchanged',all(sha(p)==value for p,value in source_before.items()))
    # Check the baseline forecasts against the earlier committed zero-intervention CSV.
    with (source/'predictions_14_buses.csv').open(encoding='utf-8-sig') as stream:old=list(csv.DictReader(stream))
    for m in MODELS:
        old_baseline=[r for r in old if r['model']==m and float(r['sigma_level'])==0]
        old_baseline.sort(key=lambda r:int(r['grid_bus']))
        check(m+'_earlier_baseline_reproduced',np.allclose([float(r['prediction_mw']) for r in old_baseline],archive[m+'.prediction_mw'][0],rtol=0,atol=1e-10))
    write_csv(out/'layer_statistics.csv',activation_rows);write_csv(out/'layer_attributions.csv',attr_rows)
    write_csv(out/'internal_ablations.csv',ablation_rows);write_csv(out/'predictions.csv',prediction_rows)
    np.savez_compressed(out/'layer_arrays.npz',**archive)
    # Compact snapshots allow repeat analysis without annual data or training.
    needed={k:snapshot[k] for k in snapshot.files if k.startswith('core.') or any(k in [m+'.weight',m+'.bias'] for m in MODELS)}
    np.savez_compressed(out/'models.npz',**needed)
    np.savez_compressed(out/'fixed_input.npz',raw=raw,mean=mean,std=std,target_pu=data['target_pu'])
    metadata=dict(scope='Fixed-input internal-layer analysis; no temperature intervention; no new training or cost solves',
        models=MODELS,stages=STAGES,layer_shapes=EXPECTED,model_parameters=20006,head_parameters=910,
        input_shape=list(raw.shape),input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),test_index=int(data['test_index']),
        captum_version=captum.__version__,torch_version=torch.__version__,
        captum_source='https://github.com/pytorch/captum/tree/v0.7.0',
        captum_commit='833d6aad27a921d2868ac0e248b4ab3ed50b13b9',
        uploaded_archive_read=False,uploaded_archive_limitation='download_file rejects uploaded captum-master.zip exceeding 32 MiB; official v0.7.0 used instead, not claimed identical.',
        attribution_method='LayerGradientXActivation; raw forecast MW per output; local, layer-specific views; no cross-layer summation or causal claim',
        internal_ablation='Set one post-ReLU channel map or one post-tanh hidden neuron to zero; recompute downstream forecast',
        total_internal_ablations=total_ablations,ablation_output_values=len(ablation_rows),
        max_gxa_error=max(manual_errors),max_affine_decomposition_error=max(head_errors),max_hidden_ablation_error=max(ablation_errors),
        source_hashes=source_before,source_code_sha256=sha(ROOT/'utils/net.py'),script_sha256=sha(__file__),
        unchanged_model_hashes=model_hashes,checks_passed=len(checks),checks_failed=0,checks=checks)
    write_json(out/'validation.json',metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['checks','source_hashes','unchanged_model_hashes']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'research/cnn_temperature_response')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'research/cnn_layer_paths')
    run(parser.parse_args())

"""V7C validation-only pilot, separate from all frozen repeat experiments.

CUDA stages refuse to start until the existing repeat queue is COMPLETE.
No test dataset is constructed or evaluated. Pilots are not five-seed results.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType

os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from source_view_flow import JointControlledFlow


def evaluate(author,model,loader,adapter):
    model.eval()
    losses,totals,count=[],{},0
    with torch.no_grad():
        for batch in loader:
            batch=tuple(x.to(author.DEVICE) for x in batch)
            prediction=adapter.forward_batch(model,batch)[0].reshape(-1)
            loss=(prediction-batch[3].reshape(-1)).square().mean()
            assert torch.isfinite(loss)
            losses.append(float(loss))
            count+=len(prediction)
            trace=model.dberta.last_trace
            for key in ('common_rank','control_norm','context_norm','safe_control_work','measured_rank','unclassified_fraction'):
                totals[key]=totals.get(key,0.)+float(trace[key].mean())*len(prediction)
            view=model.dberta.own_flow.last_view_predictions
            for code in range(7):
                totals['source_view_'+str(code+1)+'_mse']=totals.get('source_view_'+str(code+1)+'_mse',0.)+float((view[:,code]-batch[3].reshape(-1)).square().sum())
            if 'safe_first_order_drift' in trace:
                totals['first_order_drift_max']=max(totals.get('first_order_drift_max',0.),float(trace['safe_first_order_drift'].max()))
    diagnostics={key:value/count for key,value in totals.items() if key!='first_order_drift_max'}
    diagnostics['first_order_drift_max']=totals.get('first_order_drift_max')
    return float(np.mean(losses)),diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('check','run'),required=True)
    parser.add_argument('--parallel-authorization',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--feedback',choices=('none','instant','memory'),required=True)
    parser.add_argument('--rho',type=float,default=.5)
    parser.add_argument('--guard',choices=('on','off'),default='on')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=128)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    cli=parser.parse_args()
    assert cli.rho==0 and cli.feedback=='none' and cli.epochs==100 and cli.seed in (0,1,2,3)
    root=cli.baseline_root.resolve()
    cli.out=cli.out.resolve()
    assert not cli.out.exists(),'use a fresh output directory; retain previous attempts'
    if cli.device=='cuda':
        authorization=json.loads(cli.parallel_authorization.read_text())
        assert authorization['independent_gpu_authorized'] is True
        uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert uuid==authorization['gpu_uuid'],'GPU differs from deployment authorization'
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert not active,'Independent GPU is occupied; do not share training GPU'
        assert torch.cuda.is_available()
    assert shutil.disk_usage(root).free>1_200_000_000,'reserve room for a pilot checkpoint'
    cli.repo,cli.backbone,cli.data=root/'CaReFlow',root/'deberta-v3-base',root/'mosi.pkl'
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(root/'own_flow_v6'))
    sys.path.insert(0,str(cli.repo))
    helpers=importlib.import_module('run_careflow')
    control=importlib.import_module('run_control_baseline')
    adapter=importlib.import_module('encoder_adapter')
    configs=importlib.import_module('global_configs')
    configs.DEVICE=torch.device(cli.device)
    torch.set_num_threads(2)
    assert helpers.digest(cli.data)==helpers.DATA_SHA
    assert subprocess.check_output(['git','-C',str(cli.repo),'rev-parse','HEAD'],text=True).strip()==helpers.COMMIT
    author,arguments=helpers.load_author(cli)
    author.set_random_seed(cli.seed)
    train=helpers.dataset(author,cli.data,'train')
    dev=helpers.dataset(author,cli.data,'dev')
    assert (len(train),len(dev))==(1281,229)
    statistics=adapter.fit_statistics(train)
    generator=torch.Generator().manual_seed(cli.seed)
    train_loader=torch.utils.data.DataLoader(train,batch_size=32,shuffle=True,drop_last=True,generator=generator)
    dev_loader=torch.utils.data.DataLoader(dev,batch_size=128,shuffle=False)
    steps=10 if cli.stage=='check' else len(train_loader)*cli.epochs
    model,old_optimizer,old_scheduler=author.prep_for_training(steps)
    verified=helpers.pretrained_check(model,cli.backbone)
    core=model.dberta
    for name in ('reflow_a','reflow_v','reflow_a_b','reflow_v_b','rf_a','rf_v','rf_a_b','rf_v_b'):
        delattr(core,name)
    core.own_flow=JointControlledFlow(dimension=100,variant=cli.feedback,rho=cli.rho,task_guard=cli.guard=='on').to(author.DEVICE)
    adapter.install(core,statistics)
    core.to(author.DEVICE)
    core.forward=MethodType(adapter.forward_v6,core)
    del old_optimizer,old_scheduler
    optimizer,scheduler=control.optimizer_for(author,model,steps)
    files=[Path(__file__),Path(__file__).with_name('source_view_flow.py'),
           Path(__file__).with_name('native_flow.py'),root/'own_flow_v6/encoder_adapter.py',root/'run_careflow.py',root/'run_control_baseline.py']
    source={str(p.resolve()):helpers.digest(p) for p in files}
    author_files=subprocess.check_output(['git','-C',str(cli.repo),'ls-files'],text=True).splitlines()
    author_sha={n:helpers.digest(cli.repo/n) for n in author_files}
    backbone_sha={p.name:helpers.digest(p) for p in cli.backbone.iterdir() if p.is_file()}
    protocol={'name':'v8_matched_full_route_only_supervision_control_v1','stage':cli.stage,'seed':cli.seed,'epochs':cli.epochs,
              'feedback':cli.feedback,'rho':cli.rho,'guard':cli.guard,'device':cli.device,
              'parallel_authorization':authorization if cli.device=='cuda' else None,
              'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if cli.device=='cuda' else None,'transformers':importlib.import_module('transformers').__version__},
              'source_sha256':source,'author_source_sha256':author_sha,'author_commit':helpers.COMMIT,
              'backbone_sha256':backbone_sha,'data_sha256':helpers.DATA_SHA,'pretrained_tensors_verified':verified,
              'train_samples':len(train),'valid_samples':len(dev),'author_arguments':arguments,
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'flow_parameters':sum(p.numel() for p in core.own_flow.parameters()),
              'task_loss':'full source route only: (final MSE + 0.1 first-phase MSE)/1.1; seven routes still computed and evaluated',
              'selection':'author unweighted mean of validation batch MSE, batch128',
              'optimizer':{'learning_rate':1e-5,'warmup_fraction':.1,'epochs':cli.epochs,'train_batch_size':32},
              'rho_interpretation':'rho=0; no response estimator, compression or task guard installed',
              'test_policy':'no test dataset conversion, prediction, metric, or selection',
              'normalization_statistics':{m:{k:v.tolist() for k,v in values.items()} for m,values in statistics.items()},
              'limits':['Measured motion support is not identified shared semantics.',
                        'Guard protects current scalar prediction to first order.',
                        'No flow-matching endpoint coupling is used.',
                        'Matched control: only full source route receives training loss; seven routes computed for matching kernel shape. Auxiliary phase is after two of four Euler steps.', 'Relation trace placeholders are zero because no relation is measured, not evidence of zero shared response.', 'Unavailable encoded sources are zeroed before four field steps; all receivers may receive available evidence.', 'Aligned length cues remain. This is not raw-input missingness, semantic identification or a promoted five-seed candidate.']}
    cli.out.mkdir(parents=True,exist_ok=False)
    helpers.write(cli.out/'protocol.json',protocol)
    started=time.monotonic()
    history,best,best_epoch=[],float('inf'),0
    for epoch in range(1,(1 if cli.stage=='check' else cli.epochs)+1):
        model.train()
        train_losses=[]
        train_mechanism={key:[] for key in ('common_rank','control_norm','context_norm','unclassified_fraction')}
        for index,batch in enumerate(train_loader):
            batch=tuple(x.to(author.DEVICE) for x in batch)
            prediction=adapter.forward_batch(model,batch)[0].reshape(-1)
            labels=batch[3].reshape(-1)
            loss=core.last_losses['source_view_task_loss']
            assert torch.isfinite(loss)
            loss.backward()
            if cli.stage=='check':
                for parameter in (core.proj_a.weight,core.proj_v.weight,core.model.embeddings.word_embeddings.weight,core.own_flow.field.net[-1].weight):
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach()))
            for key,values in train_mechanism.items():
                values.append(float(core.last_trace[key].mean()))
            if cli.stage=='check' and index==1:
                break
        valid,diagnostics=evaluate(author,model,dev_loader,adapter)
        if valid<best:
            best,best_epoch=valid,epoch
            if cli.stage=='run':
                torch.save(model.state_dict(),cli.out/'best.pt')
        row={'epoch':epoch,'train_mse':float(np.mean(train_losses)),'valid_mse':valid,
             'training_mechanism':{key:float(np.mean(values)) for key,values in train_mechanism.items()},
             'best_epoch':best_epoch,'best_valid_mse':best,'validation_mechanism':diagnostics,
             'elapsed_seconds':time.monotonic()-started,
             'peak_gpu_bytes':torch.cuda.max_memory_allocated() if cli.device=='cuda' else 0}
        history.append(row)
        helpers.write(cli.out/'history.json',history)
        print(json.dumps(row),flush=True)
    assert source=={str(p.resolve()):helpers.digest(p) for p in files},'source changed during pilot'
    assert author_sha=={n:helpers.digest(cli.repo/n) for n in author_files}
    assert helpers.digest(cli.data)==helpers.DATA_SHA
    assert backbone_sha=={p.name:helpers.digest(cli.backbone/p.name) for p in cli.backbone.iterdir() if p.is_file()}
    if cli.stage=='check':
        helpers.write(cli.out/'checks.json',{'status':'GPU_BATCH32_AND_VALIDATION128_CHECK_PASSED' if cli.device=='cuda' else 'CPU_CHECK_PASSED',
                                           'validation_mechanism':diagnostics,'peak_gpu_bytes':row['peak_gpu_bytes'],'test_evaluated':False})
        print('V7C_CHECK_COMPLETE',flush=True)
        return
    selection={'best_epoch':best_epoch,'valid_mse':best,'epochs':cli.epochs,
               'checkpoint_sha256':helpers.digest(cli.out/'best.pt'),'protocol_sha256':helpers.digest(cli.out/'protocol.json')}
    helpers.write(cli.out/'selection.json',selection)
    model.load_state_dict(torch.load(cli.out/'best.pt',map_location='cpu'))
    author._forward_eval=adapter.forward_batch
    metrics,prediction,labels=helpers.collect(author,model,dev_loader)
    np.savez_compressed(cli.out/'validation_predictions.npz',prediction=prediction,labels=labels)
    view_predictions=[]
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            batch=tuple(x.to(author.DEVICE) for x in batch)
            adapter.forward_batch(model,batch)
            view_predictions.append(core.own_flow.last_view_predictions.cpu().numpy())
    np.savez_compressed(cli.out/'source_view_validation_predictions.npz',predictions=np.concatenate(view_predictions),labels=labels)
    helpers.write(cli.out/'results.json',{'selection':selection,'valid':metrics,'test_evaluated':False,'five_seed_result':False,'source_view_predictions_sha256':helpers.digest(cli.out/'source_view_validation_predictions.npz')})
    print('V7C_VALIDATION_PILOT_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':
    main()

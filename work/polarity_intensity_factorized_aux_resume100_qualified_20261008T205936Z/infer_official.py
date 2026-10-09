"""Selected TRAIN-only model, original encoder VAL/TEST input-only forward."""
import argparse,os,pickle,sys
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from common import sha,read,write,utc,verify
from paired_fulltrain_session_candidate import load_author_components,REMOVED_AUTHOR
from candidate_adapter import make_candidate,CandidateTail
from encoder_adapter import install
from fixed_flow_components_candidate import tensor_sha,forward_fixed,forward_batch
from train_official import source_features


def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert p['training_Release_othernode_CPU_gate'];assert p['candidate_mode'] in ('regression_aux','factorized_aux');a.out.mkdir();torch.set_num_threads(2)
    cp=a.input_root/'out/complete_final_and_selected.pt';assert sha(cp)==p['training_original_reference']['member_SHA']['out/complete_final_and_selected.pt']
    full=torch.load(cp,map_location='cpu');assert full['metadata']['candidate_mode']==p['candidate_mode'] and full['metadata']['steps']==4000;selected=full['selected_model'];assert tensor_sha(selected)==p['selected_state_SHA'];del full
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1';author,_=load_author_components(a.bundle,a.assets);author.set_random_seed(128);model,opt,scheduler=author.prep_for_training(1);del opt,scheduler
    for n in REMOVED_AUTHOR:delattr(model.dberta,n)
    model.dberta.own_flow=make_candidate(p['candidate_mode']);stats={n:{k:selected['dberta.v6_'+n+'_'+k].clone() for k in ('mean','std','active')} for n in ('audio','visual')};install(model.dberta,stats);model.dberta.forward=MethodType(forward_fixed,model.dberta);model.load_state_dict(selected,strict=True);assert tensor_sha(model.state_dict())==p['selected_state_SHA'];del selected
    tail=CandidateTail(model.dberta);model.eval().requires_grad_(False).cuda();tail.cuda();state=tensor_sha(model.state_dict());rng=torch.get_rng_state().clone();cuda=[v.clone() for v in torch.cuda.get_rng_state_all()]
    with (a.assets/'assets/mosi.pkl').open('rb') as f:data=pickle.load(f)
    result={};checks={}
    with torch.no_grad():
        for role,key in [('VAL','dev'),('TEST','test')]:
            records=data[key];ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in records];assert ids==p['official_role_IDs'][role]
            dummy=[(r[0],np.zeros((1,1),np.float32),r[2]) for r in records];ds=author.get_appropriate_dataset(dummy);on=[];off=[]
            for i in range(0,len(ids),16):
                batch=[v[i:i+16].cuda() for v in ds.tensors];prediction=forward_batch(model,batch);src,mask=source_features(model.dberta,batch);explicit,p0=tail(src,mask)
                err=float((explicit-prediction).abs().max());assert err<=1e-5
                if i==0:
                    b=[v.clone() for v in batch];b[3].fill_(7);assert torch.equal(prediction,forward_batch(model,b)) and torch.equal(prediction,forward_batch(model,batch));checks[role]=dict(dummy0vs7_and_replay_exact=True,explicit_maxerror=err)
                on.append(prediction.cpu().numpy());off.append(p0.cpu().numpy())
            result.update({role+'_ids':np.asarray(ids),role+'_prediction':np.concatenate(on),role+'_p0':np.concatenate(off)})
    assert state==tensor_sha(model.state_dict()) and torch.equal(rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
    np.savez(a.out/'fixed_official_VAL_TEST_prediction.npz',**result)
    write(a.out/'inference_result.json',dict(status='TRAIN_ONLY_SELECTED_POLARITY_INTENSITY_VAL_TEST_PREDICTION_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,selected_state_SHA=state,prediction_SHA=sha(a.out/'fixed_official_VAL_TEST_prediction.npz'),checks=checks,TRAIN_VAL_TEST_no_training_overlap=True,candidate_mode=p['candidate_mode'],peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),scalar_VAL_TEST_true_labels_not_indexed=True,all_parameters_buffers_RNG_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out','input-root'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())

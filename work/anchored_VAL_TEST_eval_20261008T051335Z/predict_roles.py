"""Fixed original best36 + fixed incremental adapter: official VAL/TEST inference."""
import argparse,os,sys,pickle,time
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from common import sha,read,write,utc,verify
from paired_fulltrain_session_candidate import load_author_components,REMOVED_AUTHOR
from anchored_flow import AnchoredFlow
from encoder_adapter import install,content_mask,encode_masked
from fixed_flow_components_candidate import tensor_sha,forward_fixed,forward_batch
from incremental_message import OriginalTail
from inference_upgrade import AnchoredMessageUpgrade
def source_features(core,batch):
    ids,visual,acoustic,dummy,mask=batch;visual=visual.squeeze(1);acoustic=acoustic.squeeze(1);valid=content_mask(mask)
    hidden=core.model(ids,attention_mask=mask)[0];text=core.LayerNorm_l(core.proj_l(hidden))*valid[...,None]
    def norm(v,n):return ((v-getattr(core,'v6_'+n+'_mean'))/getattr(core,'v6_'+n+'_std'))*getattr(core,'v6_'+n+'_active')*valid[...,None]
    audio=core.proj_a(norm(acoustic,'audio').transpose(1,2)).permute(2,0,1);vision=core.proj_v(norm(visual,'visual').transpose(1,2)).permute(2,0,1)
    audio=core.LayerNorm_a(encode_masked(core.transa,audio,valid).transpose(0,1))*valid[...,None];vision=core.LayerNorm_v(encode_masked(core.transv,vision,valid).transpose(0,1))*valid[...,None]
    return torch.stack([text,audio,vision],1),valid
def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert sha(a.parent)==p['parent']['full_SHA'];assert sha(p['adapter_remote_path'])==p['changed_state_reference']['SHA'];a.out.mkdir();torch.set_num_threads(2);start=time.monotonic()
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    full=torch.load(a.parent,map_location='cpu');selected=full['selected_model'];assert tensor_sha(selected)==p['parent']['selected_state_SHA']
    author,_=load_author_components(a.bundle,a.assets);author.set_random_seed(128);model,opt,scheduler=author.prep_for_training(1);del opt,scheduler
    for name in REMOVED_AUTHOR:delattr(model.dberta,name)
    model.dberta.own_flow=AnchoredFlow();stats={n:{k:selected['dberta.v6_'+n+'_'+k].clone() for k in ('mean','std','active')} for n in ('audio','visual')};install(model.dberta,stats);model.dberta.forward=MethodType(forward_fixed,model.dberta);model.load_state_dict(selected,strict=True);assert tensor_sha(model.state_dict())==p['parent']['selected_state_SHA'];del selected,full
    adapter=torch.load(p['adapter_remote_path'],map_location='cpu');assert adapter['updates']==940 and adapter['plan_SHA']==p['training_plan_SHA'];assert tensor_sha(adapter['adapter'])==p['adapter_state_SHA']
    tail=OriginalTail(model.dberta);assert tensor_sha(tail.state_dict())==p['tail_state_SHA'];model.dberta.own_flow=AnchoredMessageUpgrade(model.dberta.own_flow,adapter['adapter']);model.eval().requires_grad_(False).cuda();tail.cuda();del adapter
    before=tensor_sha(model.state_dict());tbefore=tensor_sha(tail.state_dict());cpu_rng=torch.get_rng_state().clone();gpu_rng=[v.clone() for v in torch.cuda.get_rng_state_all()]
    with (a.assets/'assets/mosi.pkl').open('rb') as f:whole=pickle.load(f)
    # Trusted whole pickle materializes labels. This stage indexes only inputs/IDs;
    # actual scalar VAL/TEST labels are accessed solely after preservation/replay.
    results={};cache={};checks={}
    with torch.no_grad():
        for name,key in [('VAL','dev'),('TEST','test')]:
            records=whole[key];ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in records];assert ids==p['official_role_IDs'][name]
            dummy=[(r[0],np.zeros((1,1),dtype=np.float32),s) for r,s in zip(records,ids)];dataset=author.get_appropriate_dataset(dummy);sources=[];masks=[];on=[];off=[]
            for offset in range(0,len(dataset),16):
                batch=[v[offset:offset+16].cuda() for v in dataset.tensors];source,mask=source_features(model.dberta,batch);pred=model.dberta.own_flow(source,mask,lambda z:model.dberta.predictor(model.dberta.fusion(z)))[0]
                base=tail.predictor(tail.fusion((source.sum(2)/mask.sum(1)[:,None,None]).flatten(1))).view(-1);zero=source.new_zeros((len(source),3,100));first=(source+.5*torch.stack([tail.fields[j](source[:,j],0.,zero[:,j]) for j in range(3)],1))*mask[:,None,:,None];p0=tail(first,mask,base,zero)
                if offset==0:
                    direct=forward_batch(model,batch);err=float((direct-pred).abs().max());assert err<=p['direct_forward_tolerance'];changed=[v.clone() for v in batch];changed[3].fill_(7.);assert torch.equal(direct,forward_batch(model,changed));assert torch.equal(direct,forward_batch(model,batch));checks[name]=dict(full_model_vs_explicit_original_source_maxerror=err,dummy0_vs7_exact=True,replay_exact=True,checked_rows=len(source))
                sources.append(source.cpu().numpy());masks.append(mask.cpu().numpy());on.append(pred.cpu().numpy());off.append(p0.cpu().numpy())
            results.update({name+'_ids':np.asarray(ids),name+'_prediction':np.concatenate(on),name+'_p0':np.concatenate(off)});cache.update({name+'_ids':np.asarray(ids),name+'_source':np.concatenate(sources),name+'_mask':np.concatenate(masks)})
            print('ZERO_LABEL_OFFICIAL_ROLE_COMPLETE',name,len(ids),flush=True)
    assert before==tensor_sha(model.state_dict()) and tbefore==tensor_sha(tail.state_dict());assert torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(gpu_rng,torch.cuda.get_rng_state_all()))
    np.savez(a.out/'fixed_VAL_TEST_predictions.npz',**results);np.savez(a.out/'original_source_states.npz',**cache)
    write(a.out/'prediction_result.json',dict(status='FIXED_INCREMENT_VAL229_TEST685_ZERO_LABEL_FULL_ORIGINAL_INFERENCE_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,parent=p['parent'],fixed_adapter_state_SHA=p['adapter_state_SHA'],prediction_SHA=sha(a.out/'fixed_VAL_TEST_predictions.npz'),cache_SHA=sha(a.out/'original_source_states.npz'),rows={n:len(results[n+'_ids']) for n in ('VAL','TEST')},official_role_training_overlap=p['official_role_training_overlap'],selected_state_before_wrap_SHA=p['parent']['selected_state_SHA'],upgraded_model_state_SHA=before,tail_state_SHA=tbefore,all_parameter_buffer_RNG_unchanged=True,original_full_encoder_forward=True,scalar_true_labels_indexed=False,true_labels_forwarded=False,whole_pickle_materialized=True,checks=checks,wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    print(read(a.out/'prediction_result.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','parent','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())

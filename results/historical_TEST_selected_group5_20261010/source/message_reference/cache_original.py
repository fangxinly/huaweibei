"""Cache the ORIGINAL task-finetuned first Euler states. No public-feature substitute."""
import argparse,os,sys,pickle,csv
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from common import read,write,sha,utc,verify
from paired_fulltrain_session_candidate import load_author_components,REMOVED_AUTHOR
from fold_contract import FoldGuard
from anchored_flow import AnchoredFlow
from encoder_adapter import install,content_mask,encode_masked
from fixed_flow_components_candidate import tensor_sha,forward_fixed
from incremental_message import OriginalTail
def first_state(core,batch):
    ids,visual,acoustic,dummy,mask=batch;visual=visual.squeeze(1);acoustic=acoustic.squeeze(1);valid=content_mask(mask)
    text=core.LayerNorm_l(core.proj_l(core.model(ids,attention_mask=mask)[0]))*valid[...,None]
    def norm(v,n):return ((v-getattr(core,'v6_'+n+'_mean'))/getattr(core,'v6_'+n+'_std'))*getattr(core,'v6_'+n+'_active')*valid[...,None]
    audio=core.proj_a(norm(acoustic,'audio').transpose(1,2)).permute(2,0,1);vision=core.proj_v(norm(visual,'visual').transpose(1,2)).permute(2,0,1)
    audio=core.LayerNorm_a(encode_masked(core.transa,audio,valid).transpose(0,1))*valid[...,None];vision=core.LayerNorm_v(encode_masked(core.transv,vision,valid).transpose(0,1))*valid[...,None]
    source=torch.stack([text,audio,vision],1);base=core.predictor(core.fusion((source.sum(2)/valid.sum(1)[:,None,None]).flatten(1))).view(-1)
    zero=source.new_zeros((len(source),3,100));first=(source+.5*torch.stack([core.own_flow.forward_fields[j](source[:,j],0.,zero[:,j]) for j in range(3)],1))*valid[:,None,:,None]
    slots=core.own_flow.reader(first,valid)['slots'].mean(2)
    return first,valid,base,slots
def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert sha(a.parent)==p['parent']['full_SHA'];a.out.mkdir()
    torch.set_num_threads(2);os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    old=torch.load(a.parent,map_location='cpu');selected=old['selected_model'];assert tensor_sha(selected)==p['parent']['selected_state_SHA']
    author,_=load_author_components(a.bundle,a.assets);author.set_random_seed(128);model,opt,sched=author.prep_for_training(1);del opt,sched
    for name in REMOVED_AUTHOR:delattr(model.dberta,name)
    model.dberta.own_flow=AnchoredFlow();stats={n:{k:selected['dberta.v6_'+n+'_'+k].clone() for k in ('mean','std','active')} for n in ('audio','visual')}
    install(model.dberta,stats);model.dberta.forward=MethodType(forward_fixed,model.dberta);model.load_state_dict(selected,strict=True);model.eval().requires_grad_(False).cuda();del selected,old
    assert tensor_sha(model.state_dict())==p['parent']['selected_state_SHA'];before=tensor_sha(model.state_dict());tail=OriginalTail(model.dberta).cuda();reconstructed=OriginalTail().cuda();reconstructed.load_state_dict(tail.state_dict(),strict=True)
    assert tensor_sha(tail.state_dict())==tensor_sha(reconstructed.state_dict());torch.save({'state':{k:v.cpu() for k,v in tail.state_dict().items()},'parent_full_SHA':p['parent']['full_SHA'],'parent_selected_state_SHA':before},a.out/'original_tail.pt');del reconstructed
    split=read(a.bundle/'split.json')
    with (a.assets/'assets/mosi.pkl').open('rb') as f:container=pickle.load(f)
    records=list(container['train'])+list(container['dev'])+list(container['test']);del container
    guard=FoldGuard(records,split['folds'][0]);data={};errors={};cpu_rng=torch.get_rng_state().clone();gpu_rng=[v.clone() for v in torch.cuda.get_rng_state_all()]
    with torch.no_grad():
        for role in ('fit','inner'):
            dataset=author.get_appropriate_dataset(guard.inputs(role));parts={k:[] for k in ('first','mask','base','slots','p0')}
            for start in range(0,len(dataset),16):
                batch=[v[start:start+16].cuda() for v in dataset.tensors];first,mask,base,slots=first_state(model.dberta,batch);off=tail(first,mask,base,torch.zeros_like(slots))
                for k,v in zip(parts,(first,mask,base,slots,off)):parts[k].append(v.cpu().numpy())
            for k,v in parts.items():data[role+'_'+k]=np.concatenate(v)
            data[role+'_ids']=np.asarray(guard.fold['row_ids'][role]);rows=list(csv.DictReader((a.bundle/('best36_'+role+'.csv')).open(encoding='utf8')));assert [r['row_id'] for r in rows]==data[role+'_ids'].tolist()
            # Only saved p0 column is indexed; y/p1 are not consumed at cache stage.
            expected=np.asarray([float(r['p0']) for r in rows]);errors[role]=float(np.max(abs(data[role+'_p0']-expected)));assert errors[role]<=p['cache_p0_tolerance'],errors
            print('ORIGINAL_CACHE_ROLE_COMPLETE',role,len(dataset),errors[role],flush=True)
    assert before==tensor_sha(model.state_dict());assert torch.equal(cpu_rng,torch.get_rng_state());assert all(torch.equal(x,y) for x,y in zip(gpu_rng,torch.cuda.get_rng_state_all()))
    np.savez(a.out/'original_first_states.npz',**data)
    write(a.out/'cache_result.json',dict(status='ORIGINAL_BEST36_FIRST_EULER_CACHE_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,parent=p['parent'],selected_state_SHA=before,tail_SHA=sha(a.out/'original_tail.pt'),tail_state_SHA=tensor_sha(tail.state_dict()),cache_SHA=sha(a.out/'original_first_states.npz'),cache_bytes=(a.out/'original_first_states.npz').stat().st_size,p0_saved_original_maxerror=errors,whole_pickle_materialized=True,scalar_labels_indexed=False,roles_cached=['fit','inner'],outer_inputs_not_indexed=True,parameters_buffers_rng_unchanged=True,guard_journal=guard.journal,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    print(read(a.out/'cache_result.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','parent','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())

"""Fixed original backbone, only six small donor adapters receive FIT supervision."""
import argparse,os,sys,pickle,time
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha,utc,verify
from fold_contract import FoldGuard
from fixed_flow_components_candidate import tensor_sha
from incremental_message import OriginalTail,IncrementalMessage,objective
def predict(tail,msg,c,role,device):
    out=[];off=[];context=[]
    with torch.no_grad():
        for start in range(0,len(c[role+'_ids']),32):
            v=[torch.from_numpy(c[role+'_'+k][start:start+32]).to(device) for k in ('first','mask','base','slots')];s,m,b,z=v;cx=msg(z);out.append(tail(s,m,b,cx).cpu().numpy());off.append(tail(s,m,b,torch.zeros_like(cx)).cpu().numpy());context.append(cx.cpu().numpy())
    return np.concatenate(out),np.concatenate(off),np.concatenate(context)
def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert p['real_train_enabled'];assert sha(a.cache)==p['cache_reference']['SHA'];assert sha(a.tail)==p['tail_reference']['SHA'];a.out.mkdir()
    torch.set_num_threads(2);torch.manual_seed(128);torch.cuda.manual_seed_all(128);started=time.monotonic();c=dict(np.load(a.cache,allow_pickle=False));tail=OriginalTail();frozen=torch.load(a.tail,map_location='cpu');tail.load_state_dict(frozen['state'],strict=True);assert frozen['parent_full_SHA']==p['parent']['full_SHA'];tail.cuda();frozen_sha=tensor_sha(tail.state_dict());msg=IncrementalMessage().cuda();params=[x for x in msg.parameters() if x.requires_grad];opt=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    split=read(a.bundle/'split.json')
    with (a.assets/'assets/mosi.pkl').open('rb') as f:whole=pickle.load(f)
    records=list(whole['train'])+list(whole['dev'])+list(whole['test']);del whole
    guard=FoldGuard(records,split['folds'][0]);fit=guard.supervision('fit');assert c['fit_ids'].tolist()==guard.fold['row_ids']['fit'];y=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in fit],dtype=np.float32);assert np.isfinite(y).all() and (abs(y)<=3).all();del fit,records,guard
    data={k:torch.from_numpy(c['fit_'+k]).cuda() for k in ('first','mask','base','slots')};target=torch.from_numpy(y).cuda();order_rng=np.random.default_rng(128);orders=[];history=[];updates=0
    init_p,init_off,_=predict(tail,msg,c,'fit','cuda');assert np.array_equal(init_p,init_off);assert np.max(abs(init_off-c['fit_p0']))<=p['cache_p0_tolerance']
    for epoch in range(1,21):
        order=order_rng.permutation(len(y));orders.append(order);running=[]
        for start in range(0,len(y),32):
            idx=torch.as_tensor(order[start:start+32],device='cuda');opt.zero_grad(set_to_none=True);cx=msg(data['slots'][idx]);prediction=tail(data['first'][idx],data['mask'][idx],data['base'][idx],cx);loss,parts=objective(prediction,target[idx],cx);loss.backward()
            assert torch.isfinite(loss) and all(x.grad is not None and torch.isfinite(x.grad).all() for x in params);assert all(x.grad is None for x in tail.parameters());norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step();updates+=1;running.append((len(idx),float(loss.detach()),float(parts['huber']),float(parts['context_square'])))
        values=np.asarray(running);r=dict(epoch=epoch,updates=updates,train_objective=float(np.average(values[:,1],weights=values[:,0])),train_huber=float(np.average(values[:,2],weights=values[:,0])),context_square=float(np.average(values[:,3],weights=values[:,0])));history.append(r);print(r,flush=True)
    assert updates==940 and frozen_sha==tensor_sha(tail.state_dict());msg.eval();state_sha=tensor_sha(msg.state_dict());pred={}
    for role in ('fit','inner'):
        prediction,off,context=predict(tail,msg,c,role,'cuda');assert np.max(abs(off-c[role+'_p0']))<=p['cache_p0_tolerance'];pred.update({role+'_prediction':prediction,role+'_p0':off,role+'_ids':c[role+'_ids'],role+'_context':context})
    pred['adapter_state_SHA']=np.asarray(state_sha);np.savez(a.out/'fixed_final_predictions.npz',**pred)
    np.savez(a.out/'FIT_training_labels.npz',row_ids=c['fit_ids'],y=y)
    def cpu(v):
        if torch.is_tensor(v):return v.detach().cpu()
        if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return type(v)(cpu(x) for x in v)
        return v
    torch.save(dict(adapter=cpu(msg.state_dict()),optimizer=cpu(opt.state_dict()),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),order_rng=order_rng.bit_generator.state,orders=np.stack(orders),history=history,updates=updates,plan_SHA=a.plan_sha,parent=p['parent'],cache=p['cache_reference'],tail=p['tail_reference'],adapter_state_SHA=state_sha,composite_reconstruction_not_standalone=True),a.out/'complete_changed_training_state.pt')
    write(a.out/'train_result.json',dict(status='ORIGINAL_BEST36_MESSAGE_INCREMENT_FIXED20_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,updates=updates,epochs=20,fit_rows=len(y),inner_input_rows=len(c['inner_ids']),inner_labels_not_indexed=True,no_epoch_selection=True,parent=p['parent'],cache_reference=p['cache_reference'],tail_reference=p['tail_reference'],unchanged_tail_state_SHA=frozen_sha,adapter_state_SHA=state_sha,trainable_parameters=sum(x.numel() for x in params),history=history,prediction_SHA=sha(a.out/'fixed_final_predictions.npz'),changed_state_SHA=sha(a.out/'complete_changed_training_state.pt'),composite_state_not_standalone=True,wall_seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    print(read(a.out/'train_result.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','cache','tail','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())

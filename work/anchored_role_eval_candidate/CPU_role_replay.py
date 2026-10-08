import argparse,os,sys
from pathlib import Path
import numpy as np
import torch
from common import sha,read,write,utc,verify
from incremental_message import OriginalTail,IncrementalMessage
from fixed_flow_components_candidate import tensor_sha
def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets)
    for path,key in ((a.cache,'role_cache_reference'),(a.prediction,'role_prediction_reference'),(a.tail,'tail_reference'),(a.state,'changed_state_reference')):assert sha(path)==p[key]['SHA'],key
    a.out.mkdir();torch.set_num_threads(2);c=dict(np.load(a.cache,allow_pickle=False));pred=dict(np.load(a.prediction,allow_pickle=False));tail=OriginalTail();tail.load_state_dict(torch.load(a.tail,map_location='cpu')['state'],strict=True);assert tensor_sha(tail.state_dict())==p['tail_state_SHA'];state=torch.load(a.state,map_location='cpu');assert state['updates']==940 and state['plan_SHA']==p['training_plan_SHA'];msg=IncrementalMessage();msg.load_state_dict(state['adapter'],strict=True);msg.eval();assert tensor_sha(msg.state_dict())==p['adapter_state_SHA']
    before=tensor_sha(tail.state_dict());mbefore=tensor_sha(msg.state_dict());rng=torch.get_rng_state().clone();errors={}
    for name in ('VAL','TEST'):
        assert pred[name+'_ids'].tolist()==c[name+'_ids'].tolist()==p['official_role_IDs'][name];on=[];off=[]
        with torch.no_grad():
            for start in range(0,len(pred[name+'_ids']),32):
                source=torch.from_numpy(c[name+'_source'][start:start+32]);mask=torch.from_numpy(c[name+'_mask'][start:start+32]);base=tail.predictor(tail.fusion((source.sum(2)/mask.sum(1)[:,None,None]).flatten(1))).view(-1);zero=source.new_zeros((len(source),3,100));first=(source+.5*torch.stack([tail.fields[j](source[:,j],0.,zero[:,j]) for j in range(3)],1))*mask[:,None,:,None];slots=tail.reader(first,mask)['slots'].mean(2);on.append(tail(first,mask,base,msg(slots)).numpy());off.append(tail(first,mask,base,zero).numpy())
        actual=np.concatenate(on);p0=np.concatenate(off);errors[name]=dict(new=float(np.max(abs(actual-pred[name+'_prediction']))),p0=float(np.max(abs(p0-pred[name+'_p0']))));assert max(errors[name].values())<=p['CPU_replay_tolerance'],errors
    assert before==tensor_sha(tail.state_dict()) and mbefore==tensor_sha(msg.state_dict()) and torch.equal(rng,torch.get_rng_state())
    write(a.out/'CPU_role_replay_result.json',dict(status='VAL_TEST_ORIGINAL_SOURCE_FULL_FLOW_MESSAGE_CPU_REPLAY_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,rows={'VAL':229,'TEST':685},prediction_SHA=sha(a.prediction),errors=errors,parameters_buffers_RNG_unchanged=True,labels_not_indexed=True,CPU_encoder_forward=False));print(read(a.out/'CPU_role_replay_result.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','cache','tail','state','prediction','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())

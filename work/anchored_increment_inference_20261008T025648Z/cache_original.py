import argparse,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from common import sha,read,write,utc
from anchored_flow import AnchoredFlow
from incremental_message import OriginalTail,IncrementalMessage
from inference_upgrade import AnchoredMessageUpgrade
from fixed_flow_components_candidate import tensor_sha
def check(p,bundle,assets,device):
    torch.set_num_threads(2);torch.manual_seed(128)
    assert sha(Path(p['parent']['remote_path']))==p['parent']['full_SHA'];parent=torch.load(p['parent']['remote_path'],map_location='cpu');selected=parent['selected_model'];assert tensor_sha(selected)==p['parent']['selected_state_SHA']
    flow=AnchoredFlow();flow.load_state_dict({k[len('dberta.own_flow.'):]:v for k,v in selected.items() if k.startswith('dberta.own_flow.')},strict=True);del selected,parent
    state=torch.load(p['changed_state_remote_path'],map_location='cpu');assert sha(p['changed_state_remote_path'])==p['changed_state_reference']['SHA'];tail=OriginalTail();tail.load_state_dict(torch.load(p['tail_remote_path'],map_location='cpu')['state'],strict=True)
    model=AnchoredMessageUpgrade(flow,state['adapter']).to(device);tail.to(device);model.eval();before=tensor_sha(model.state_dict());rng=torch.get_rng_state().clone();grng=[v.clone() for v in torch.cuda.get_rng_state_all()] if device=='cuda' else []
    x=torch.randn(8,3,13,100,device=device);mask=torch.ones(8,13,dtype=torch.bool,device=device);mask[0,1:]=False;mask[1,7:]=False
    # Random source tests full forward integration only; it is not a new real-data score.
    with torch.no_grad():
        source=x*mask[:,None,:,None];base=tail.predictor(tail.fusion((source.sum(2)/mask.sum(1)[:,None,None]).flatten(1))).view(-1);zero=source.new_zeros((8,3,100));first=(source+.5*torch.stack([flow.to(device).forward_fields[j](source[:,j],0.,zero[:,j]) for j in range(3)],1))*mask[:,None,:,None]
        slots=flow.reader(first,mask)['slots'].mean(2);expected=tail(first,mask,base,model.adapter(slots));decoder=lambda z:tail.predictor(tail.fusion(z));actual=model(x,mask,decoder)[0];error=float((actual-expected).abs().max());assert error<=2e-6
        dummy=model(x,mask,decoder,torch.full((8,),7.,device=device))[0];assert torch.equal(actual,dummy)
        masked=x.clone();masked[~mask[:,None,:,None].expand_as(masked)]=999.;assert torch.equal(actual,model(masked,mask,decoder)[0])
        perm=torch.arange(7,-1,-1,device=device);assert torch.allclose(model(x[perm],mask[perm],decoder)[0],actual[perm],atol=2e-6,rtol=2e-6)
    assert before==tensor_sha(model.state_dict());return dict(device=device,saved_trained_adapter_full_original_flow_formula_equivalence_maxerror=error,dummy_label_and_padding_invariance=True,permutation_invariance=True,parameters_buffers_unchanged=True,no_real_data_or_encoder_forward=True)
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out','parent'):a.add_argument('--'+n,type=Path,required=True)
    a.add_argument('--plan-sha',required=True);x=a.parse_args();p=read(x.plan);assert sha(x.plan)==x.plan_sha;x.out.mkdir();d='cuda' if p['check_device']=='cuda' else 'cpu';r=check(p,x.bundle,x.assets,d);write(x.out/'integration_result.json',dict(actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,check=r));print(r)

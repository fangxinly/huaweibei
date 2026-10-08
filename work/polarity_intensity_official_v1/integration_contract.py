"""Synthetic optimizer/tail export test; not a full encoder or dataset test."""
import datetime,io,json,os,sys
from pathlib import Path
import torch
from torch import nn
from candidate_adapter import make_candidate,optimizer_groups,candidate_objective,CandidateTail
import train_official,infer_official

class SyntheticCore(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.own_flow=make_candidate(mode)
        self.fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100))
        self.predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1))
    def forward(self,x,mask):
        return self.own_flow(x,mask,lambda h:self.predictor(self.fusion(h)))[0]

class SyntheticModel(nn.Module):
    def __init__(self,mode):
        super().__init__();self.dberta=SyntheticCore(mode)

def run():
    torch.set_num_threads(2)
    modes={}
    for mode in ('regression_aux','factorized_aux'):
        torch.manual_seed(128)
        model=SyntheticModel(mode);groups,named=optimizer_groups(model)
        opt=torch.optim.AdamW(groups)
        x=torch.randn(7,3,8,100);mask=torch.ones(7,8,dtype=torch.bool);mask[0,4:]=False
        y=torch.tensor([-2.,-1.,-.1,0.,.1,1.,2.])
        for _ in range(3):
            model.train();opt.zero_grad(set_to_none=True)
            p=model.dberta(x,mask);loss=candidate_objective(model,p,y);loss.backward()
            assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in named)
            opt.step()
        model.eval();tail=CandidateTail(model.dberta)
        before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
        with torch.no_grad():
            p=model.dberta(x,mask);explicit,p0=tail(x,mask)
            assert torch.equal(p,explicit) and torch.equal(p0,model.dberta.own_flow.last_off['prediction'])
        assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        serialized=io.BytesIO();torch.save(dict(model=model.state_dict(),tail=tail.state_dict(),optimizer=opt.state_dict(),mode=mode),serialized)
        serialized.seek(0);saved=torch.load(serialized,map_location='cpu')
        restored=SyntheticModel(saved['mode']);restored.load_state_dict(saved['model'],strict=True);restored.eval()
        restored_tail=CandidateTail(restored.dberta);restored_tail.load_state_dict(saved['tail'],strict=True)
        with torch.no_grad():
            assert torch.equal(p,restored.dberta(x,mask))
            assert torch.equal(p,restored_tail(x,mask)[0]) and torch.equal(p0,restored_tail(x,mask)[1])
        modes[mode]=dict(three_synthetic_updates=True,all_optimized_gradients_finite=True,
                         tail_ON_OFF_exact=True,serialization_replay_exact=True,parameters=sum(v.numel() for v in model.parameters()),
                         optimizer_parameters_unique_complete=True,full_encoder_or_real_data=False)
    assert modes['regression_aux']['parameters']==modes['factorized_aux']['parameters']
    return dict(status='NATIVE_SYNTHETIC_POLARITY_INTENSITY_INTEGRATION_COMPLETE',
        actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        torch=torch.__version__,device='CPU',real_dataset_or_weights=False,new_VAL_TEST_scores=False,modes=modes,
        original_full_training_and_inference_modules_imported=True,
        candidate_full_encoder_precheck_D_B_qualification=False)

if __name__=='__main__':
    if len(sys.argv)!=2:raise ValueError('Explicit output JSON required')
    result=run();Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

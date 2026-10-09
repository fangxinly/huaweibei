"""Native CPU continuation trajectory test using OfficialUpgrade and stochastic inputs."""
import argparse,copy,datetime,hashlib,json,os,random,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.bundle))
import numpy as np
import torch
from torch import nn
from candidate_adapter import make_candidate,optimizer_groups
from polarity_intensity_flow import train_objective
from fixed_flow_components_candidate import tensor_sha
class M(nn.Module):
 def __init__(self):
  super().__init__();self.dberta=nn.Module();self.dberta.own_flow=make_candidate(mode);self.decoder=nn.Linear(300,1)
 def forward(self,x):return self.dberta.own_flow(x,torch.ones(x.shape[0],x.shape[2],dtype=torch.bool),self.decoder)[0]
def makeopt(m):
 groups,_=optimizer_groups(m)
 o=torch.optim.AdamW(groups);s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:step/400 if step<400 else max(0,(4000-step)/3600));return o,s
a.out.mkdir(exist_ok=False)
for mode in ('regression_aux','factorized_aux'):
 random.seed(128);np.random.seed(128);torch.manual_seed(128);torch.set_num_threads(2);original=M();o,s=makeopt(original)
 def rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
 def reset(r):random.setstate(r[0]);np.random.set_state(r[1]);torch.set_rng_state(r[2])
 def step(m,o,s):
  random.random();np.random.random();x=torch.randn(4,3,5,100);y=torch.randn(4);o.zero_grad(set_to_none=True);pred=m(x);loss=train_objective(m.dberta.own_flow,y,'TRAIN')[0];loss.backward();o.step();s.step();return pred.detach().clone()
 for i in range(16):step(original,o,s)
 path=a.out/('toy_prefix_'+mode+'.pt');torch.save(dict(model=original.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),rng=rng()),path)
 expected=[]
 for i in range(16,40):expected.append(step(original,o,s))
 expectedSHA=tensor_sha(original.state_dict());expectedRNG=rng();resumed=M();ro,rs=makeopt(resumed);state=torch.load(path,map_location='cpu');resumed.load_state_dict(state['model'],strict=True);ro.load_state_dict(state['optimizer']);rs.load_state_dict(state['scheduler']);reset(state['rng']);assert rs.last_epoch==16
 for i in range(16,40):assert torch.equal(step(resumed,ro,rs),expected[i-16])
 assert tensor_sha(resumed.state_dict())==expectedSHA and rs.last_epoch==40
 assert expectedRNG[0]==rng()[0] and all(np.array_equal(x,y) for x,y in zip(expectedRNG[1],rng()[1])) and torch.equal(expectedRNG[2],rng()[2])
 for old,new in zip(o.state_dict()['state'].values(),ro.state_dict()['state'].values()):
  for k in ('step','exp_avg','exp_avg_sq'):assert torch.equal(old[k],new[k])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=dict(status='POLARITY_INTENSITY_BOTH_MODES_PREFIX16_RESUME_TO40_CPU_TRAJECTORY_IDENTICAL',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),source_SHA=sha(__file__),candidate_SHA=sha(a.bundle/'resume_official_mse100_v1.py'),qualified_modes=['regression_aux','factorized_aux'],prefix_counted_once=True,prediction_parameter_Adam_scheduler_python_numpy_torch_RNG_identical=True,real_TRAIN=False,final_steps=40)
(a.out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

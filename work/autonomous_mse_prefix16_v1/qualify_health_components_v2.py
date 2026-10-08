"""CPU qualification: instrumentation must leave the actual optimizer trajectory identical."""
import argparse,copy,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.bundle))
import torch
from torch import nn
from torch.nn import functional as F
from official_upgrade import OfficialUpgrade
from training_health_components_v2 import component_gradients,before_optimizer_step,after_optimizer_step,fixed_sample_indices
from fixed_flow_components_candidate import tensor_sha

class Model(nn.Module):
 def __init__(self):
  super().__init__();self.dberta=nn.Module();self.dberta.own_flow=OfficialUpgrade();self.decoder=nn.Linear(300,1)
 def forward(self,x,mask):return self.dberta.own_flow(x,mask,self.decoder)[0]

torch.set_num_threads(2)
for device in ('cpu','cuda'):
 for size in (1,511,512,513,16777218,98304000):
  idx=fixed_sample_indices(size,device);assert idx.dtype==torch.long and int(idx.min())==0 and int(idx.max())==size-1 and len(idx)==min(size,512)
 probe=torch.arange(16777218,dtype=torch.float32,device=device)
 idx=fixed_sample_indices(len(probe),device);assert torch.isfinite(probe[idx]).all()
 del probe,idx
torch.cuda.synchronize()
torch.manual_seed(128);left=Model();right=copy.deepcopy(left)
def opt(m):
 groups=[dict(params=[p for n,p in m.named_parameters() if p.requires_grad and not n.endswith('gain')],lr=1e-5,weight_decay=.01),dict(params=[m.dberta.own_flow.gain],lr=.001,weight_decay=0.)]
 o=torch.optim.AdamW(groups);s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:step/400 if step<400 else max(0,(4000-step)/3600));return o,s
lo,ls=opt(left);ro,rs=opt(right);x=torch.randn(4,3,5,100);mask=torch.ones(4,5,dtype=torch.bool);y=torch.linspace(-2,2,4)
rows=[]
for step in range(16):
 left.train();right.train();lo.zero_grad(set_to_none=True);ro.zero_grad(set_to_none=True)
 state=torch.get_rng_state().clone();pl=left(x,mask);torch.set_rng_state(state);pr=right(x,mask);assert torch.equal(pl,pr)
 tl=F.mse_loss(pl,y);rl=.01*left.dberta.own_flow.last_context.square().mean()
 rng=torch.get_rng_state().clone();components=component_gradients(left,tl,rl)
 assert torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in left.parameters())
 (tl+rl).backward();(F.mse_loss(pr,y)+.01*right.dberta.own_flow.last_context.square().mean()).backward()
 assert all(torch.equal(p.grad,q.grad) for p,q in zip(left.parameters(),right.parameters()) if p.requires_grad)
 snap=before_optimizer_step(left,lo);lo.step();ls.step();ro.step();rs.step();updates=after_optimizer_step(left,snap)
 assert tensor_sha(left.state_dict())==tensor_sha(right.state_dict())
 rows.append(dict(step=step+1,components=components,updates=updates))
# Fresh zero-gradient AdamW must match a rounded pure-decay update exactly.
fresh=Model();fo=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
for parameter in fresh.parameters():
 if parameter.requires_grad:parameter.grad=torch.zeros_like(parameter)
snap=before_optimizer_step(fresh,fo);fo.step();decay=after_optimizer_step(fresh,snap)
assert all(v['beyond_decay_values']==0 for v in decay.values())
a.out.mkdir(exist_ok=False)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
result=dict(status='HEALTH_V2_NATIVE_CPU_SYNTHETIC_TRAJECTORY_IDENTICAL',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),source_SHA=sha(__file__),helper_SHA=sha(a.bundle/'training_health_components_v2.py'),first16_same_parameter_trajectory=True,gradient_RNG_no_accumulation=True,rounded_pure_decay_control_passed=True,real_TRAIN=False,GPU_large_index_qualification_only=True,large_index_bounds_and_real_gather_passed=True,steps=rows,decay_control=decay)
(a.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('steps','decay_control')}),flush=True)

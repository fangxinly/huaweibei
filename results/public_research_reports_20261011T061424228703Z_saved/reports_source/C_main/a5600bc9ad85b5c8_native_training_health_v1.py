"""Original optimizer's synthetic startup; no real encoder/data/VAL/TEST."""
import copy,datetime,json,os,sys
from pathlib import Path
import torch
from torch import nn
from official_upgrade import OfficialUpgrade
from training_health_v1 import gradient_summary,before_update,after_update,original_task_objective

class Core(nn.Module):
    def __init__(self):
        super().__init__();self.own_flow=OfficialUpgrade();self.decoder=nn.Linear(300,1)

class Model(nn.Module):
    def __init__(self):super().__init__();self.dberta=Core()
    def forward(self,x,mask):return self.dberta.own_flow(x,mask,self.dberta.decoder)[0]

def run():
    torch.set_num_threads(2);torch.manual_seed(128)
    initial=Model();x=torch.randn(7,3,8,100);mask=torch.ones(7,8,dtype=torch.bool);mask[0,4:]=False
    y=torch.tensor([-2.,-1.,-.1,0.,.1,1.,2.]);results={}
    for kind in ('huber1','mse'):
        model=copy.deepcopy(initial);flow=model.dberta.own_flow
        gain=flow.gain;message=[p for p in flow.message.parameters() if p.requires_grad];ids={id(p) for p in message}|{id(gain)}
        ordinary=[(n,p) for n,p in model.named_parameters() if p.requires_grad and id(p) not in ids]
        nd=('bias','LayerNorm.bias','LayerNorm.weight')
        opt=torch.optim.AdamW([dict(params=[p for n,p in ordinary if not any(k in n for k in nd)],lr=1e-5,weight_decay=.01),
                              dict(params=[p for n,p in ordinary if any(k in n for k in nd)],lr=1e-5,weight_decay=0.),
                              dict(params=[gain],lr=.001,weight_decay=0.),dict(params=message,lr=.001,weight_decay=.01)])
        schedule=lambda step:float(step)/400 if step<400 else max(0.,float(4000-step)/3600)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
        records=[]
        for step in range(1,11):
            model.train();opt.zero_grad(set_to_none=True);p=model(x,mask)
            loss=original_task_objective(flow,p,y,kind=kind,role='TRAIN');loss.backward()
            gradients=gradient_summary(model);before=before_update(model);learning_rates=[g['lr'] for g in opt.param_groups]
            opt.step();updates=after_update(model,before);scheduler.step()
            if step==1:
                assert all(gradients[g]['gradient_RMS']==0 for g in ('forward_fields','reader','role_head','message'))
                assert gradients['gain']['gradient_RMS']>0
            records.append(dict(step=step,loss=float(loss.detach()),gain_tanh=float(flow.gain.detach().tanh()),gradients=gradients,updates=updates,learning_rates_before_update=learning_rates,
                                prediction_minus_pooled_base_RMS=float((p.detach()-flow.last_base.detach()).square().mean().sqrt())))
        for role in ('VAL','TEST','INNER'):
            try:original_task_objective(flow,p,y,kind=kind,role=role)
            except ValueError:pass
            else:raise AssertionError('Non-TRAIN labels accepted')
        results[kind]=records
    return dict(status='NATIVE_SYNTHETIC_ORIGINAL_FLOW_TRAINING_HEALTH_COMPLETE',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pid=os.getpid(),fullargv=[sys.executable]+sys.argv,torch=torch.__version__,device='CPU',real_data_or_weights=False,new_VAL_TEST_scores=False,
        encoder_not_instantiated=True,original_zero_gain_first_step_block_verified=True,original_400_warmup_4000_schedule=True,results=results,
        scientific_scope='Synthetic startup and instrumentation only; not persistent real-training evidence or loss-performance comparison')

if __name__=='__main__':
    if len(sys.argv)!=2:raise ValueError('Explicit output JSON required')
    result=run();Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)

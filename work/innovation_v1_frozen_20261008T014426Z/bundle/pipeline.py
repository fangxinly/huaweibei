"""Whole-pipeline OOF on task-training videos; separate untouched calibration videos."""
import hashlib, io
import numpy as np
import torch
from contract import groups,video,affine,residual_slope,residual_gate,Reducer,PAIRS
from models import Baseline,ConditionalFlow,ConditionalRegression,Proposal,GateHead,gate_features

def seed(n):torch.manual_seed(n);np.random.seed(n);torch.cuda.manual_seed_all(n) if torch.cuda.is_available() else None
def tensor(x,device):return torch.as_tensor(x,dtype=torch.float32,device=device)
def state_sha(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        x=v.detach().cpu().contiguous();h.update(name.encode());h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()
def subset(features,idx):return [x[idx] for x in features]
def fit(model,loss_fn,n,steps,device,seed_value):
    model.to(device).train();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01);rng=np.random.default_rng(seed_value);tg=torch.Generator(device=device).manual_seed(seed_value)
    history=[];initial=state_sha(model)
    for step in range(steps):
        idx=rng.choice(n,size=min(64,n),replace=False);opt.zero_grad();loss=loss_fn(idx,tg)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();history.append(float(loss.detach().cpu()))
        if (step+1)%40==0 or step+1==steps:print(dict(module=type(model).__name__,step=step+1,budget=steps,loss=history[-1]),flush=True)
    model.eval().requires_grad_(False)
    # Complete small-head optimizer, random state and order generator snapshots.
    complete=dict(initial_SHA=initial,final_SHA=state_sha(model),model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer=opt.state_dict(),steps=steps,history=history,numpy_order_rng=rng.bit_generator.state,
        noise_rng=tg.get_state().cpu(),torch_rng=torch.get_rng_state().clone(),cuda_rng=[v.cpu() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    return complete

def fit_baseline(features,y,ids,cfg,device,serial,records):
    reducer=Reducer(features,cfg['latent_dim']);z=tensor(reducer.raw_transform(features),device);yt=tensor(y,device);seed(128+serial);model=Baseline(input_dim=z.shape[1])
    saved=fit(model,lambda idx,g:(model(z[idx])-yt[idx]).square().mean(),len(y),cfg['baseline_steps'],device,128+serial)
    records.append(dict(kind='baseline',serial=serial,train_ids=list(map(str,ids)),reducer=reducer.state(),complete=saved))
    return reducer,model
def baseline_oof(features,y,ids,cfg,device,serial,records):
    q=np.full(len(y),np.nan);splits=groups(ids,cfg['inner_crossfit_k'],128)
    for j,held in enumerate(splits):
        train=np.setdiff1d(np.arange(len(y)),held);red,model=fit_baseline(subset(features,train),y[train],ids[train],cfg,device,serial+j,records)
        assert not {video(r) for r in ids[train]}&{video(r) for r in ids[held]}
        with torch.no_grad():q[held]=model(tensor(red.raw_transform(subset(features,held)),device)).cpu().numpy()
        records[-1]['prediction_held_ids']=list(map(str,ids[held]));records[-1]['oof_prediction']=q[held].copy()
    assert np.isfinite(q).all();return q

class Stack:
    def __init__(self,features,y,ids,cfg,device,serial,records):
        self.device=device;self.cfg=cfg;self.ids=set(map(str,ids));self.records=records
        raw_oof=baseline_oof(features,y,ids,cfg,device,serial*100,records);self.alpha,self.beta=affine(raw_oof,y)
        self.reducer,self.baseline=fit_baseline(features,y,ids,cfg,device,serial*100+90,records)
        z=tensor(self.reducer.transform(features),device);target=tensor(y-(self.alpha*raw_oof+self.beta),device)
        self.decompositions={};self.proposals={}
        for name,factory in [('flow',ConditionalFlow),('regression',ConditionalRegression)]:
            seed(512+serial);model=factory(cfg['latent_dim']).to(device)
            saved=fit(model,lambda idx,g:model.objective(z[idx],g),len(y),cfg['flow_steps'],device,512+serial)
            records.append(dict(kind=name,serial=serial,train_ids=list(map(str,ids)),complete=saved))
            self.decompositions[name]=model
        donor=torch.stack([z[:,n] for m,n in PAIRS],1)
        with torch.no_grad():
            messages={'full':donor,'flow':donor-self.decompositions['flow'].means(z),'regression':donor-self.decompositions['regression'].means(z)}
        for name in ('full','flow','regression'):
            seed(1024+serial);proposal=Proposal(cfg['latent_dim']).to(device);message=messages[name]
            saved=fit(proposal,lambda idx,g:(proposal(z[idx],message[idx])[0]-target[idx]).square().mean(),len(y),cfg['proposal_steps'],device,1024+serial)
            records.append(dict(kind='proposal_'+name,serial=serial,train_ids=list(map(str,ids)),complete=saved,
                residual_target_source='baseline whole-video OOF; affine on this stack training OOF only',baseline_calibration=(self.alpha,self.beta)))
            self.proposals[name]=proposal
    def predict(self,features):
        with torch.no_grad():
            z=tensor(self.reducer.transform(features),self.device);p0=self.alpha*self.baseline(tensor(self.reducer.raw_transform(features),self.device))+self.beta
            donor=torch.stack([z[:,n] for m,n in PAIRS],1);means={k:m.means(z) for k,m in self.decompositions.items()}
            messages={'full':donor,'flow':donor-means['flow'],'regression':donor-means['regression']};result={'p0':p0.cpu().numpy().astype(float)}
            for name in messages:
                delta,channels=self.proposals[name](z,messages[name]);phi=gate_features(z,p0,delta,channels,messages[name]);result[name]=dict(delta=delta.cpu().numpy().astype(float),phi=phi.cpu().numpy())
            result['conditional_donor_MSE']={k:float((v-donor).square().mean()) for k,v in means.items()}
        return result
    def snapshot(self):return dict(reducer=self.reducer.state(),alpha=self.alpha,beta=self.beta,baseline=self.baseline.state_dict(),decompositions={k:m.state_dict() for k,m in self.decompositions.items()},proposals={k:m.state_dict() for k,m in self.proposals.items()})

class Mechanism:
    def __init__(self,train_x,train_y,train_ids,cal_x,cal_y,cal_ids,cfg,device='cpu'):
        self.cfg=cfg;self.device=device;self.records=[]
        if {video(i) for i in train_ids}&{video(i) for i in cal_ids}:raise ValueError('Training/calibration video overlap')
        n=len(train_y);oof={'p0':np.full(n,np.nan)}
        for name in ('full','flow','regression'):oof[name]=dict(delta=np.full(n,np.nan),phi=None)
        self.lineage=[]
        for j,held in enumerate(groups(train_ids,cfg['outer_crossfit_k'],128)):
            train=np.setdiff1d(np.arange(n),held);start=len(self.records)
            stack=Stack(subset(train_x,train),train_y[train],train_ids[train],cfg,device,j+1,self.records);p=stack.predict(subset(train_x,held));oof['p0'][held]=p['p0']
            held_v={video(r) for r in train_ids[held]}
            # Enforce that every normalization/base/flow/proposal predecessor excludes the held videos.
            for rec in self.records[start:]:assert not {video(r) for r in rec['train_ids']}&held_v
            self.lineage.append(dict(held_ids=list(map(str,train_ids[held])),record_start=start,record_end=len(self.records)))
            for name in ('full','flow','regression'):
                oof[name]['delta'][held]=p[name]['delta']
                if oof[name]['phi'] is None:oof[name]['phi']=np.empty((n,p[name]['phi'].shape[1]),dtype=np.float32)
                oof[name]['phi'][held]=p[name]['phi']
        assert np.isfinite(oof['p0']).all();rho=train_y-oof['p0'];self.gates={}
        for name in ('flow','regression'):
            x=tensor(oof[name]['phi'],device);yt=tensor(rho,device);seed(2048);gate=GateHead(cfg['latent_dim']).to(device)
            saved=fit(gate,lambda idx,g:(gate(x[idx])-yt[idx]).square().mean(),n,cfg['gate_steps'],device,2048)
            self.records.append(dict(kind='residual_gate_'+name,train_ids=list(map(str,train_ids)),complete=saved));self.gates[name]=gate
        phi=tensor(oof['flow']['phi'],device);d=tensor(oof['flow']['delta'],device);yt=tensor(rho,device);seed(2048);self.ordinary=GateHead(cfg['latent_dim']).to(device)
        saved=fit(self.ordinary,lambda idx,g:(torch.sigmoid(self.ordinary(phi[idx]))*d[idx]-yt[idx]).square().mean(),n,cfg['gate_steps'],device,2048)
        self.records.append(dict(kind='ordinary_task_gate',train_ids=list(map(str,train_ids)),complete=saved))
        self.oof=oof;self.stack=Stack(train_x,train_y,train_ids,cfg,device,99,self.records);cal=self.stack.predict(cal_x);cal_rho=np.asarray(cal_y)-cal['p0'];self.slope={}
        with torch.no_grad():
            for name in self.gates:
                rh=self.gates[name](tensor(cal[name]['phi'],device)).cpu().numpy();self.slope[name]=residual_slope(rh,cal_rho)
            og=torch.sigmoid(self.ordinary(tensor(cal['flow']['phi'],device))).cpu().numpy();od=og*cal['flow']['delta']
        den=float(od@od);self.ordinary_scale=0. if den<1e-12 else max(0.,float(od@cal_rho/den))
        self.calibration=dict(cal_ids=list(map(str,cal_ids)),slopes=self.slope,ordinary_scale=self.ordinary_scale,
            note='CAL videos excluded from all baselines, decomposition, proposals, gate fitting and normalization. They fit only final one-dimensional slopes.')
        for rec in self.records:assert not {video(r) for r in rec['train_ids']}&{video(r) for r in cal_ids}
    def predict(self,features):
        p=self.stack.predict(features);p0=p['p0'];result={'baseline_calibrated':p0,'full_message':p0+p['full']['delta']};extra={}
        with torch.no_grad():
            for name in self.gates:
                rh=self.gates[name](tensor(p[name]['phi'],self.device)).cpu().numpy().astype(float);g=residual_gate(rh,p[name]['delta'],self.slope[name]);result[name+'_residual_gate']=p0+g*p[name]['delta'];extra[name]=dict(rhat=rh,delta=p[name]['delta'],gate=g)
            og=np.clip(self.ordinary_scale*torch.sigmoid(self.ordinary(tensor(p['flow']['phi'],self.device))).cpu().numpy().astype(float),0.,1.)
        result['flow_ordinary_gate']=p0+og*p['flow']['delta'];extra['conditional_donor_MSE']=p['conditional_donor_MSE']
        assert all(np.isfinite(v).all() for v in result.values())
        return result,extra
    def complete(self):return dict(config=self.cfg,records=self.records,stack=self.stack.snapshot(),gates={k:m.state_dict() for k,m in self.gates.items()},ordinary=self.ordinary.state_dict(),calibration=self.calibration,oof=self.oof,lineage=self.lineage)
    @classmethod
    def restore(cls,saved,device='cpu'):
        obj=object.__new__(cls);obj.cfg=saved['config'];obj.device=device;obj.records=saved['records'];obj.oof=saved['oof'];obj.lineage=saved['lineage'];obj.calibration=saved['calibration'];obj.slope=obj.calibration['slopes'];obj.ordinary_scale=obj.calibration['ordinary_scale']
        st=object.__new__(Stack);st.device=device;st.cfg=obj.cfg;st.reducer=object.__new__(Reducer);st.reducer.stats=saved['stack']['reducer'];st.alpha=saved['stack']['alpha'];st.beta=saved['stack']['beta']
        st.baseline=Baseline(input_dim=sum(len(v[0]) for v in st.reducer.stats)).to(device);st.baseline.load_state_dict(saved['stack']['baseline']);st.baseline.eval().requires_grad_(False)
        st.decompositions={};st.proposals={}
        for name,factory in [('flow',ConditionalFlow),('regression',ConditionalRegression)]:
            m=factory(obj.cfg['latent_dim']).to(device);m.load_state_dict(saved['stack']['decompositions'][name]);st.decompositions[name]=m.eval().requires_grad_(False)
        for name in ('full','flow','regression'):
            m=Proposal(obj.cfg['latent_dim']).to(device);m.load_state_dict(saved['stack']['proposals'][name]);st.proposals[name]=m.eval().requires_grad_(False)
        obj.stack=st;obj.gates={}
        for name in ('flow','regression'):
            m=GateHead(obj.cfg['latent_dim']).to(device);m.load_state_dict(saved['gates'][name]);obj.gates[name]=m.eval().requires_grad_(False)
        obj.ordinary=GateHead(obj.cfg['latent_dim']).to(device);obj.ordinary.load_state_dict(saved['ordinary']);obj.ordinary.eval().requires_grad_(False)
        return obj

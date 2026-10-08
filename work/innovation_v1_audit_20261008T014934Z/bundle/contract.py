"""Video-isolated contracts for frozen-feature mechanism v1."""
import hashlib, json
from pathlib import Path
import numpy as np

PAIRS=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))  # receiver, donor
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def video(row):return str(row).split('[')[0]
def groups(ids,k=5,seed=128):
    ids=np.asarray(ids).astype(str);vs=sorted(set(map(video,ids)));assert len(vs)>=k
    counts={v:sum(video(r)==v for r in ids) for v in vs};bins=[[] for _ in range(k)];sizes=[0]*k
    def tie(v):return hashlib.sha256(f'{seed}:{v}'.encode()).hexdigest()
    for v in sorted(vs,key=lambda v:(-counts[v],tie(v))):
        j=min(range(k),key=lambda j:(sizes[j],len(bins[j]),j));bins[j].append(v);sizes[j]+=counts[v]
    result=[np.array([i for i,r in enumerate(ids) if video(r) in set(b)]) for b in bins]
    assert sorted(np.concatenate(result).tolist())==list(range(len(ids)))
    return result
def split_calibration(ids,fraction=.2,seed=128):
    ids=np.asarray(ids).astype(str);vs=sorted(set(map(video,ids)),key=lambda v:hashlib.sha256(f'cal:{seed}:{v}'.encode()).hexdigest())
    assert len(vs)>=12;prefix=np.cumsum([sum(video(r)==v for r in ids) for v in vs[:-1]])
    stop=int(np.argmin(abs(prefix-fraction*len(ids))))+1;cal=set(vs[:stop]);a=np.array([i for i,r in enumerate(ids) if video(r) not in cal]);b=np.array([i for i,r in enumerate(ids) if video(r) in cal])
    assert not {video(ids[i]) for i in a}&{video(ids[i]) for i in b}
    return a,b
def affine(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);xc=x-x.mean();yc=y-y.mean()
    if xc@xc<1e-12:raise ValueError('Degenerate calibration base')
    a=float(xc@yc/(xc@xc));b=float(y.mean()-a*x.mean());return a,b
def residual_slope(rhat,rho):
    rhat=np.asarray(rhat,float);rho=np.asarray(rho,float);den=float(rhat@rhat)
    return 0. if den<1e-12 else max(0.,float(rhat@rho/den))
def residual_gate(rhat,delta,slope):
    # Exact zero effect: accept nothing; no epsilon-defined risk-bound claim.
    q=np.zeros_like(np.asarray(delta,float));np.divide(slope*np.asarray(rhat,float),delta,out=q,where=abs(delta)>1e-12)
    return np.clip(q,0.,1.)
def utility(rho,delta,gate):
    d=np.asarray(gate)*np.asarray(delta);return 2*np.asarray(rho)*d-d*d

class Reducer:
    """Each fit uses only its declared training rows; no label/role access."""
    def __init__(self,features,dim=32):
        self.stats=[]
        for x in features:
            x=np.asarray(x,dtype=np.float64);mu=x.mean(0);std=x.std(0);active=std>=1e-6;std=np.maximum(std,1e-6)
            z=(x-mu)/std*active
            # A task-free fixed orientation is shared by all OOF models. Fitted
            # PCA axes would rotate gate inputs between OOF and final models.
            rng=np.random.default_rng(128+x.shape[1]);basis=np.linalg.qr(rng.normal(size=(x.shape[1],min(dim,x.shape[1]))))[0]
            if basis.shape[1]<dim:basis=np.pad(basis,((0,0),(0,dim-basis.shape[1])))
            # Fixed scaling prevents a high-dimensional text PCA from dominating.
            latent=z@basis;scale=np.maximum(latent.std(0),1e-6)
            self.stats.append((mu,std,active,basis,scale))
    def transform(self,features):
        return np.stack([(((np.asarray(x,float)-m)/s*a)@b/q).astype(np.float32) for x,(m,s,a,b,q) in zip(features,self.stats)],axis=1)
    def state(self):return self.stats
    def raw_transform(self,features):
        return np.concatenate([((np.asarray(x,float)-m)/s*a).astype(np.float32) for x,(m,s,a,b,q) in zip(features,self.stats)],1)

def synthetic_contract():
    ids=np.array([f'v{v}[{i}]' for v in range(25) for i in range(3)])
    for held in groups(ids):
        train=np.setdiff1d(np.arange(len(ids)),held);assert not {video(ids[i]) for i in train}&{video(ids[i]) for i in held}
    a,b=split_calibration(ids);assert len(a)+len(b)==len(ids)
    rho=np.array([0.,1.,-2.,.5]);d=np.array([0.,2.,-1.,-1.]);g=residual_gate(rho,d,1.)
    assert np.array_equal(g,np.array([0.,.5,1.,0.]))
    assert np.max(abs(utility(rho,d,g)-(rho*rho-(rho-g*d)**2)))<1e-12
    x=np.arange(10.);alpha,beta=affine(x,1.3*x-.2);assert abs(alpha-1.3)<1e-12 and abs(beta+.2)<1e-12
    assert residual_slope(x,-x)==0
    return dict(video_partitions_disjoint=True,zero_delta_safe=True,utility_identity=True,affine_recovery=True,negative_calibration_slope_falls_back=True)

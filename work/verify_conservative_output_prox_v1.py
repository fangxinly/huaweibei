"""Algebraic numerical witnesses, no dataset or model access."""
from pathlib import Path
import datetime,json,numpy as np
rng=np.random.default_rng(91819);r=rng.uniform(-3,3,10000);eps=rng.uniform(0,3,10000);cap=rng.uniform(.001,3,10000)
d=-np.sign(r)*np.minimum(np.maximum(np.abs(r)-eps,0),cap)
u=lambda t:t*t+2*r*t+2*eps*np.abs(t)
assert np.all(np.abs(d)<=cap) and np.max(u(d))<=1e-12
grid=np.linspace(-1,1,401)
smallest=(u(grid[:,None]*cap[None,:])-u(d)[None,:]).min()
assert smallest>=-1e-12
delta=rng.uniform(-4,4,10000)
direct=delta*delta+2*r*delta+2*eps*np.abs(delta)<0
piece=(r*delta<0)&(np.abs(r)>eps)&(np.abs(delta)>0)&(np.abs(delta)<2*(np.abs(r)-eps))
assert np.array_equal(direct,piece)
bound_cases=np.array([-2.,-1.,0.,1.,2.]);e=np.abs(bound_cases);z=-np.sign(bound_cases)*np.maximum(np.abs(bound_cases)-e,0);assert np.array_equal(z,np.zeros(5))
report={'status':'OUTPUT_SCALE_CONSERVATIVE_QUADRATIC_SOFT_THRESHOLD_AND_ACCEPTANCE_INTERVAL_NUMERICAL_WITNESSES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':91819,'rows':10000,'grid_points_per_row':401,'minimum_grid_minus_analytic_objective':float(smallest),'maximum_analytic_objective':float(np.max(u(d))),'acceptance_boolean_mismatches':int(np.count_nonzero(direct!=piece)),'dataset_or_model_read':False,'scope':'Output-scale proxy only. Does not solve the nonlinear feasible message problem or establish an empirical conditional-mean error guarantee.'}
out=Path(__file__).resolve().parent.parent/'outputs'/'原消息相对保守接受输出解析核验.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))

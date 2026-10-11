from pathlib import Path
import ast,datetime,hashlib,json,numpy as np
r=Path(__file__).resolve().parents[1];j=Path('D:/CodexBackups/selective_flow_20261003_1105/label_free_jacobian_20261005T1312Z');cache=Path('D:/CodexBackups/selective_flow_20261003_1105/soft_teacher_cache_20261005T1246Z/teacher_cache_v1');receipt=json.loads((j/'jacobian_receipt.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert receipt['source_sha256']==sha(r/'work/collect_label_free_jacobian_v1.py') and not receipt['labels_requested'] and not receipt['teacher_targets_requested'] and not receipt['test_requested']
tree=ast.parse((r/'work/collect_label_free_jacobian_v1.py').read_text());keys={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='cache' and isinstance(n.slice,ast.Constant)};assert keys=={'state','mask','old_context','reference_prediction'}
reports={}
for split,n in [('train',1281),('dev',229)]:
 p=j/f'{split}_jacobian.npz';m=receipt['splits'][split];assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes']
 a=np.load(p);b=np.load(cache/f'{split}_cache.npz');assert set(a.files)=={'reference_jacobian','reference_prediction','row_id'}
 assert np.array_equal(a['row_id'],np.arange(n)) and a['reference_jacobian'].shape==(n,3,100)
 assert all(np.isfinite(a[k]).all() for k in a.files)
 norm=np.linalg.norm(a['reference_jacobian'].reshape(n,-1),axis=1);assert norm.min()>0
 error=float(np.max(np.abs(a['reference_prediction']-b['reference_prediction'])));assert error<2e-5
 report={'rows':n,'jacobian_norm_min':float(norm.min()),'jacobian_norm_max':float(norm.max()),'reference_replay_max_error':error,'jacobian_sha256':sha(p)}
 if split=='train':
  factor=2*(a['reference_prediction']-b['y'])[:,None,None]*a['reference_jacobian'];maxerror=float(np.max(np.abs(factor-b['teacher_gradient'])));assert maxerror<2e-6
  report['train_loss_gradient_factorization_max_error']=maxerror
  scale=np.load(cache/'train_gradient_rms.npy');v=a['reference_jacobian'].astype(float)/scale[None,:,None];target=b['teacher_gradient'].astype(float)/scale[None,:,None]
  rng=np.random.RandomState(916);pred=rng.normal(size=target.shape);norm2=np.sum(v*v,axis=(1,2),keepdims=True);projected=np.sum(pred*v,axis=(1,2),keepdims=True)/norm2*v
  ptarget=np.sum(target*v,axis=(1,2),keepdims=True)/norm2*v
  assert np.max(np.abs(np.sum((pred-projected)*v,axis=(1,2))))<1e-8
  error_target=float(np.max(np.abs(ptarget-target)));assert error_target<.005
  assert np.mean((projected-ptarget)**2)<=np.mean((pred-ptarget)**2)+1e-10
  report['normalized_teacher_target_off_subspace_max_error']=error_target
  report['orthogonal_projection_synthetic_error_nonincrease']=True
 reports[split]=report
out={'status':'LABEL_FREE_JACOBIAN_SHA_FIELDS_AST_AND_TRAIN_FACTORIZATION_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt':receipt,'cache_access_AST':sorted(keys),'reports':reports,'scope':'TRAIN labels only in independent factorization audit of existing teacher; collector never requested y. DEV no label factorization performed here; prediction replay only. Does not certify learned residual sign or task improvement.'}
(r/'outputs/标签无关Jacobian采集与梯度分解独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(reports))

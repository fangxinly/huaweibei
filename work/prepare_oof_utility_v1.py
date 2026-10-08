from pathlib import Path
import ast,datetime,hashlib,json,shutil
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
D=Path('D:/CodexBackups/selective_flow_20261003_1105')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage(D).free>1024**3
now=datetime.datetime.now(datetime.timezone.utc)
stamp=now.strftime('%Y%m%dT%H%M%SZ')
local=D/('oof_utility_preparation_'+stamp);local.mkdir()
remote='/data/coding/group_teacher_v1_20261005T1650Z/oof_utility_diagnostic_v1_'+stamp
original=D/'group_teacher_oof_20261006T0308Z/train_scalar_oof.npz'
assert sha(original)=='cff1a52f7e828c96cacf070ae4e051a5c34042ebb9435f44f5b3242cad91a7ea'
with np.load(original,allow_pickle=False) as z:
    assert np.array_equal(z['row_ids'],np.arange(1281))
    assert z['mu'].shape==(1281,) and np.isfinite(z['mu']).all()
    np.savez_compressed(local/'mu_input.npz',row_ids=z['row_ids'],fold=z['fold'],mu=z['mu'])
with np.load(local/'mu_input.npz',allow_pickle=False) as z:
    assert set(z.files)=={'row_ids','fold','mu'}
    with np.load(original,allow_pickle=False) as orig:
        assert all(np.array_equal(z[k],orig[k]) for k in z.files)
source=HERE/'diagnose_oof_teacher_utility_v1.py'
ast.parse(source.read_text(encoding='utf-8'))
tree=ast.parse(source.read_text(encoding='utf-8'))
npfunc=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='new_path'][0]
assert [a.arg for a in npfunc.args.args]==['batch','target']
assert not any(isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Constant) and x.slice.value=='y' for x in ast.walk(npfunc))
plan=json.loads((HERE/'train_oracle_plan_v2.json').read_text(encoding='utf-8'))
pins=plan['pinned_files'].copy()
helper='/data/coding/train_oracle_diagnostic_20261006T014605Z/diagnose_train_oracle_v2.py'
pins[helper]=plan['diagnostic_source_sha256']
oldarrays='/data/coding/train_oracle_diagnostic_20261006T014605Z/out/diagnostics.npz'
pins[oldarrays]='92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
pins[remote+'/mu_input.npz']=sha(local/'mu_input.npz')
capture='/data/coding/soft_vector_research_20261005T1220Z/capture_soft_vector_v19.py'
pins[capture]='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
new=dict(status='FROZEN_NEW_OOF_TEACHER_UTILITY_DIAGNOSTIC',utc=now.isoformat(),new_root=remote,
    base_root=plan['base_root'],formal_root=plan['formal_root'],mapping_path=plan['mapping_path'],
    helper_source=helper,old_diagnostic_arrays=oldarrays,source_sha256=sha(source),pinned_files=pins,
    expected_uuid=plan['expected_uuid'],beta=plan['controller']['beta'],rows=1281,videos=52,
    maximum_seconds=2700,preservation_reserve_seconds=7200,minimum_remote_free_bytes=1024**3,
    maximum_peak_allocated_bytes=1024**3,estimated_lease_end_utc=plan['estimated_lease_end_utc'],
    platform_lease_end_verified=False,parameter_updates=0,optimizer_steps=0,
    generator='Original immutable solve, y=None; rho=p0-muOOF; original three .25 updates and .25 message trust',
    selector='Earliest minimum pure (prediction-muOOF)^2 in float64; includes original F, excludes proximity penalty from selection',
    labels='Mu-only input file has row/fold/mu, no y. Cache y first read only after prediction NPZ written and hash frozen. No DEV/TEST.',
    original_input_replay_scope='Fresh coordinate original-F and learned control precheck versus immutable Oracle prediction arrays; full original-input C2 strict disk replay is pinned completed evidence, not a new fullmodel forward.',
    capture_coverage='New subroot is inside teacher tree captured by immutable capture19; source/plan/NPZ/log/receipt/inventory and exit JSON must be audited after actual capture.',
    limits='Single-seed fixed C2 best37 and reference fullTRAINfit/DEVselected. OOF scalar label isolation does not remove reference leakage. New finite-path diagnostic, no new fitted controller, strict upper bound, whole-pipeline crossfit, TEST or generalization/SOTA claim.',
    original_OOF_npz_sha256=sha(original),mu_input_sha256=sha(local/'mu_input.npz'))
(local/'plan.json').write_text(json.dumps(new,indent=2),encoding='utf-8')
shutil.copy2(source,local/source.name)
manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in local.iterdir() if p.is_file()}
proof=dict(utc=now.isoformat(),local=str(local),remote=remote,manifest=manifest,original_OOF_sha256=sha(original),
    label_free_scalar_exact=True,AST_solver_no_y=True,source_frozen_not_yet_deployed=True,
    fresh_local_disk_free={drive:shutil.disk_usage(drive+':/').free for drive in ['C','D']})
(BASE/'outputs/OOF教师效用短实验冻结与输入审核.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof))

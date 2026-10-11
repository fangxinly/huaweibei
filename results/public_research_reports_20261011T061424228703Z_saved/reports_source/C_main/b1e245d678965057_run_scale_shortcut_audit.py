"""Freeze and execute one local saved-array diagnostic, then seal originals on D."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
ap=argparse.ArgumentParser();ap.add_argument('--clock',required=True);a=ap.parse_args()
base=Path('D:/CodexBackups/selective_flow_20261003_1105');stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
root=base/('scale_shortcut_audit_'+stamp);assert not root.exists();root.mkdir();source=root/'source';source.mkdir()
old=base/'inner_lovo_output_gate_20261007T160409Z/source';op=read(old/'protocol.json')
new=base/'fixed_weak_donor_mask_20261007T161226Z/A_original_capsule';nm=read(new/'manifest.json')
provenance=read('outputs/正式CaReFlow比较来源与预算本地准备接续.json')
author=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/outputs/careflow_reproduction/official/model_reflow_new.py')
assert sha(author)==provenance['source_files_checked']['model_reflow_new.py']['expected_sha256']
refs={}
for model,inputs in [('best20',old),('best36',new/'original')]:
    for role in ('fit','inner'):
        for suffix in ('bfp.npz','evaluation_targets.npy'):
            name=role+'_'+suffix;path=inputs/name
            expected=op['input_sha256'][name] if model=='best20' else nm['member_sha256']['original/'+name]
            assert sha(path)==expected
            dest=model+'_'+role+('_bfp.npz' if suffix=='bfp.npz' else '_targets.npy');shutil.copy2(path,source/dest);refs[dest]=dict(path=str(path),SHA=expected)
        if model=='best36':
            name=role+'_donor_masks.npz';path=inputs/name;expected=nm['member_sha256']['original/'+name];assert sha(path)==expected
            dest='best36_'+name;shutil.copy2(path,source/dest);refs[dest]=dict(path=str(path),SHA=expected)
for p,name in [(author,'pinned_author_model_reflow_new.py'),(Path('work/weak_retention_train40_20261007T145201Z/bundle/anchored_flow.py'),'pinned_anchored_flow.py'),(Path('work/weak_retention_train40_20261007T145201Z/bundle/fixed_flow_components_candidate.py'),'pinned_fixed_flow_components.py'),(Path('work/sentiment_metrics_careflow_v1.py'),'sentiment_metrics_careflow_v1.py'),(Path('work/scale_shortcut_diagnostic.py'),'scale_shortcut_diagnostic.py')]:
    shutil.copy2(p,source/name);refs[name]=dict(path=str(p),SHA=sha(p))
assert sha(source/'sentiment_metrics_careflow_v1.py')=='a91e46693d8a690e20240a39232c9b63287d7c7fc6148711bb474e6b6e9b99a6'
plan=dict(status='SAVED_SCALE_SHORTCUT_FIT_INNER_PROTOCOL_FROZEN',actualclock_freeze_UTC=a.clock,scope='Fixed selected20 and36 FIT1494/INNER264 saved predictions only; no forward, new training, TEST/OUTER files, reselecting or deployment. INNER already selected both checkpoints; exploratory.',
    rows=dict(fit=1494,inner=264),selected_state_SHA=dict(best20=op['parent_selected_state_SHA'],best36=nm['original_receipt_sha256']),
    regressions='Within-role ordinary least squares with intercept, p~b, delta~b, rho~b, f~b; both delta and rho residualized for partial correlation. R2 descriptive, not information-content measure.',
    calibration='Primary unweighted segment OLS y=alpha*b+beta using FIT only. Secondary equal-video OLS. Also FIT-only p affine calibration for both fixed checkpoints. All reported, no choosing a winner.',
    ratio_threshold=.3,ratio_std_ddof=0,bootstrap_draws=10000,bootstrap_seed=20261008,bootstrap_scope='Fixed model/calibration conditional paired video resampling; 11 INNER videos; no calibration refit/epoch selection in bootstrap.',
    source_refs=refs,worker_SHA=sha(source/'scale_shortcut_diagnostic.py'),member_sha256={p.name:sha(p) for p in source.iterdir()})
receipt=read(new/'original/actual_stage_receipt.json');plan['selected_state_SHA']['best36']=receipt['selected_state_sha256'];assert sha(new/'original/actual_stage_receipt.json')==nm['original_receipt_sha256']
write(source/'protocol.json',plan);psha=sha(source/'protocol.json')
cmd=[sys.executable,'-X','utf8',str((source/'scale_shortcut_diagnostic.py').resolve()),'--plan',str((source/'protocol.json').resolve()),'--plan-sha',psha,'--out',str((root/'actual').resolve())]
write(root/'actual_dispatch.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),fullargv=cmd))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf8')
with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
    child=subprocess.Popen(cmd,stdout=out,stderr=err,env=env);code=child.wait()
write(root/'natural_exit.json',dict(pid=child.pid,fullargv=cmd,natural_exit=code,actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA=psha))
shutil.copy2(__file__,root/Path(__file__).name)
if code: print((root/'stderr.log').read_text(encoding='utf8'));raise SystemExit(code)
files={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file()};write(root/'member_SHA.json',files)
zip_path=root/'complete_local_originals.zip'
with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED,6) as z:
    for p in root.rglob('*'):
        if p.is_file() and p!=zip_path:z.write(p,str(p.relative_to(root)).replace('\\','/'))
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,h in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
write(root/'actual_local_archive_audit.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ZIP_SHA=sha(zip_path),ZIP_CRC=True,unique_members=True,all_member_SHA=True,local_saved_array_CPU_only=True))
pointer=dict(status='SCALE_SHORTCUT_SAVED_ARRAY_AUDIT_COMPLETE',D=str(root),protocol_SHA=psha,zip_SHA=sha(zip_path),result_SHA=sha(root/'actual/actual_result.json'),overall_research_complete=False)
write('work/scale_shortcut_current.json',pointer);print(json.dumps(pointer));print((root/'stdout.log').read_text(encoding='utf8'))

"""Seal the already naturally completed worker; repair only interpreter-option representation."""
import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path('D:/CodexBackups/selective_flow_20261003_1105/inner_lovo_output_gate_20261007T160409Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
n=json.loads((root/'natural_exit.json').read_text(encoding='utf8'));r=json.loads((root/'actual/actual_stage_receipt.json').read_text(encoding='utf8'))
assert n['natural_exit']==0 and n['pid']==r['pid'] and n['protocol_SHA']==r['plan_sha256']
cmd=n['fullargv'];assert cmd[1:3]==['-X','utf8'];assert r['fullargv']==[cmd[0]]+cmd[3:]
assert sha(root/'source/protocol.json')==r['plan_sha256']
plan=json.loads((root/'source/protocol.json').read_text(encoding='utf8'))
for k,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha(root/'source'/k)==h
shutil.copy2('work/freeze_run_inner_lovo_gate.py',root/'original_dispatcher_failed_only_postrun_argv_assert.py')
shutil.copy2(__file__,root/Path(__file__).name)
write(root/'postrun_argv_representation_repair.json',dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 original_sealing_failure='Natural0 worker completed, dispatcher postrun assertion compared process argv including -X utf8 with sys.argv excluding interpreter options.',
 both_original_argv_retained=True,only_allowed_comparison_normalization='Remove the original explicit -X utf8 interpreter option pair from process argv, preserving every worker argument.',
 no_worker_or_fitting_rerun=True,original_natural0_child=n['pid'],source_protocol_inputs_unchanged=True))
manifest={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file()}
write(root/'local_original_manifest.json',dict(actual_capture_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),member_sha256=manifest,source='Actual local CPU run, not remote GPU capture'))
with zipfile.ZipFile(root/'complete_local_CPU_capture.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(root)).replace('\\','/'))
with zipfile.ZipFile(root/'complete_local_CPU_capture.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for k,h in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==h
write(root/'capture_receipt.json',dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=sha(root/'complete_local_CPU_capture.zip'),bytes=(root/'complete_local_CPU_capture.zip').stat().st_size,CRC_unique_all_member_SHAs_passed=True))
pointer=dict(D=str(root),protocol_SHA=r['plan_sha256'],capture_SHA=sha(root/'complete_local_CPU_capture.zip'),status=r['status'])
write(Path('work/inner_lovo_output_gate_current.json'),pointer);print(json.dumps(pointer))

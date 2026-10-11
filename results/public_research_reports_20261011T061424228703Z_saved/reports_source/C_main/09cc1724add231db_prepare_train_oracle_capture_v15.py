"""Prepare additive capture coverage locally. No upload or remote execution."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil
work=Path(__file__).resolve().parent; outputs=work.parent/'outputs'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
plan=json.loads((work/'train_oracle_plan_v2.json').read_text(encoding='utf-8'))
root=plan['new_root']
source=(work/'capture_soft_vector_v14.py').read_text(encoding='utf-8').replace('capture_soft_vector_v14.py','capture_soft_vector_v15.py')
insertion='''
 oracle=Path(ORACLE_ROOT_LITERAL)
 tree(oracle,'train_oracle')
 if oracle.exists():
  for filename in ['execution.log','wrapper.log']:
   if (oracle/filename).is_file():add(oracle/filename,'train_oracle/'+filename)
  matches=[]
  for candidate in sorted(Path('/proc').iterdir()):
   if not candidate.name.isdigit():continue
   try:
    raw=(candidate/'cmdline').read_bytes().decode().split('\\0')[:-1]
   except (OSError,UnicodeError):continue
   if any(Path(arg).name=='diagnose_train_oracle_v2.py' for arg in raw) and str(oracle/'train_oracle_plan_v2.json') in raw:
    matches.append({'pid':int(candidate.name),'argv':raw})
  oi=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(oracle),gpu=inv['gpu'],
          compute=q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),
          matching_processes=matches,data_disk_free=shutil.disk_usage('/data').free,
          receipt_present=(oracle/'out/receipt.json').is_file(),progress_present=(oracle/'out/progress.json').is_file(),
          scope='Capture evidence, not proof of diagnostic completion; original receipt and arrays need independent audit.')
  encoded=json.dumps(oi,indent=2).encode();z.writestr('train_oracle_inventory.json',encoded)
  members['train_oracle_inventory.json']={'bytes':len(encoded),'sha256':sha(encoded)}
'''.replace('ORACLE_ROOT_LITERAL',repr(root))
anchor=" z.writestr('large_file_manifest.json',json.dumps(large,indent=2));"
assert source.count(anchor)==1
source=source.replace(anchor,insertion+anchor)
target=work/'capture_soft_vector_v15.py';assert not target.exists()
ast.parse(source);target.write_text(source,encoding='utf-8')
auditor=(work/'audit_soft_snapshot_v14.py').read_text(encoding='utf-8').replace('capture_soft_vector_v14.py','capture_soft_vector_v15.py')
target_audit=work/'audit_soft_snapshot_v15.py';assert not target_audit.exists()
ast.parse(auditor);target_audit.write_text(auditor,encoding='utf-8')
utc=datetime.datetime.now(datetime.timezone.utc);stamp=utc.strftime('%Y%m%dT%H%M%SZ')
proof=dict(utc=utc.isoformat(),status='LOCAL_ADDITIVE_CAPTURE15_PREPARED_NOT_DEPLOYED',
    oracle_root=root,capture_sha256=sha(target),old_root_auditor_sha256=sha(target_audit),
    diagnostic_auditor_sha256=sha(work/'audit_train_oracle_v1.py'),
    plan_sha256=sha(work/'train_oracle_plan_v2.json'),original_capture14_unchanged=True,
    source_ast_passed=True,remote_uploaded=False,gpu_executed=False,snapshot_executed=False,
    contract='Old root audit via audit_soft_snapshot_v15 plus existing finite/C2 auditors; new original oracle receipt and1281arrays via audit_train_oracle_v1. No completion inferred from tree or inventory.')
(outputs/'TRAIN_Oracle新增捕获覆盖准备核验.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
state=outputs/'研究接续状态.md'
with state.open('a',encoding='utf-8') as f:
    f.write(f"\nUTC{utc.isoformat()}：本地已准备加法capture15覆盖冻结Oracle新root、source/plan/out数组/receipt/progress/日志与匹配进程argv；配套旧根audit15语法通过，原capture14和旧冻结科学源不改。未部署/未新capture/未GPU，不当保存完成。证据：TRAIN_Oracle新增捕获覆盖准备核验.json。审批阻塞无新恢复证据，未重试。\n")
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')
free=shutil.disk_usage(dest).free;assert free>100*1024**2
dest=dest/('train_oracle_capture_preparation_'+stamp);dest.mkdir()
manifest={}
for path in [target,target_audit,Path(__file__),outputs/'TRAIN_Oracle新增捕获覆盖准备核验.json',state]:
    copy=dest/path.name;shutil.copy2(path,copy);assert sha(path)==sha(copy)
    manifest[path.name]=dict(bytes=copy.stat().st_size,sha256=sha(copy))
(dest/'preservation_manifest.json').write_text(json.dumps(dict(utc=utc.isoformat(),fresh_d_free_bytes=free,scope='Local preparation only; no remote capture',files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=proof['status'],files=len(manifest),destination=str(dest)),ensure_ascii=True))

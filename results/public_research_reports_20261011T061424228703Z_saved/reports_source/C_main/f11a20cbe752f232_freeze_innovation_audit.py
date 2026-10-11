import argparse,ast,datetime,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path('work/innovation_v1').resolve()))
from contract import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args();stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
c=read('work/innovation_v1_current.json');D=Path(c['D_actual']);v=read(D/'A_train/actual_D_verification.json');assert v['natural_exit']==0
root=Path('work')/('innovation_v1_audit_'+stamp);root.mkdir();b=root/'bundle';b.mkdir();source=Path(c['bundle'])
original=read(source/'execution_protocol.json')
for name,h in original['source_sha256'].items():assert sha(source/name)==h;shutil.copy2(source/name,b/name)
for name in ('innovation_cpu_audit.py','innovation_audit_wrapper.py'):
    ast.parse(Path('work',name).read_text(encoding='utf8'));shutil.copy2(Path('work',name),b/name)
plan={k:original[k] for k in ('GPU_UUID','runtime_versions','config','methods','conservative_lease_end_UTC','remote_space_floor_bytes')}
plan.update(status='INDEPENDENT_NATIVE_CPU_AUDIT_FROZEN',actualclock_UTC=a.clock,source_sha256={x.name:sha(x) for x in b.iterdir()},parent_execution_plan_SHA=c['execution_plan_SHA'],train_archive_SHA=v['archive_SHA'],cache_archive_SHA=read(D/'A_cache/actual_D_verification.json')['archive_SHA'],CPU_prediction_tolerance=1e-4,tolerance_predeclared_before_any_CPU_replay=True,labels_enabled=False,encoder_CPU_forward=False)
write(b/'audit_protocol.json',plan);zpath=root/'audit_source_bundle.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,6) as z:
    for x in b.iterdir():z.write(x,'bundle/'+x.name)
shutil.copy2(zpath,D/zpath.name);write(D/'CPU_audit_source_seal.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_SHA=sha(zpath),copied_SHA=sha(D/zpath.name),protocol_SHA=sha(b/'audit_protocol.json'),CPU_replay_not_yet_executed=True))
pointer=dict(root=str(root.resolve()),bundle=str(b.resolve()),ZIP_SHA=sha(zpath),plan_SHA=sha(b/'audit_protocol.json'));write('work/innovation_v1_audit_current.json',pointer);print(pointer)

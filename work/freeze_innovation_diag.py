import argparse,ast,datetime,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path('work/innovation_v1').resolve()))
from contract import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args();stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
c=read('work/innovation_v1_audit_current.json');old=Path(c['bundle']);plan=read(old/'audit_protocol.json');D=Path(read('work/innovation_v1_current.json')['D_actual'])
root=Path('work')/('innovation_v1_train_diag_'+stamp);root.mkdir();b=root/'bundle';b.mkdir()
for name,h in plan['source_sha256'].items():assert sha(old/name)==h;shutil.copy2(old/name,b/name)
src=Path('work/innovation_saved_train_diagnostic.py');ast.parse(src.read_text(encoding='utf8'));shutil.copy2(src,b/src.name)
wrapper=(old/'innovation_audit_wrapper.py').read_text(encoding='utf8').replace("a.plan.parent/'innovation_cpu_audit.py'","a.plan.parent/'innovation_saved_train_diagnostic.py'")
(b/'innovation_saved_diag_wrapper.py').write_text(wrapper,encoding='utf8');ast.parse(wrapper)
plan.update(status='SAVED_TRAIN_CAL_DIAGNOSTIC_FROZEN_NO_INNER_TARGETS',actualclock_UTC=a.clock,source_sha256={x.name:sha(x) for x in b.iterdir()},labels_enabled='only already saved original FIT supervision; TRAIN and CAL indices',no_training=True,no_INNER_labels=True)
write(b/'diagnostic_protocol.json',plan);zp=root/'diagnostic_source_bundle.zip'
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
    for x in b.iterdir():z.write(x,'bundle/'+x.name)
shutil.copy2(zp,D/zp.name);pointer=dict(root=str(root.resolve()),ZIP_SHA=sha(zp),plan_SHA=sha(b/'diagnostic_protocol.json'));write('work/innovation_v1_diag_current.json',pointer);print(pointer)

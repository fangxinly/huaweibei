from pathlib import Path
import json,zipfile,hashlib,datetime,ast,shutil

w=Path(__file__).parent
root=w/'group_teacher_plan_20261005T1650Z'
archive=Path('D:/CodexBackups/selective_flow_20261003_1105/new_p4_assets_20261005T1220Z/common_assets.zip')
def digest_stream(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def sha(path):
 with Path(path).open('rb') as f:return digest_stream(f)

plan=json.loads((root/'group_teacher_plan_v1.json').read_text(encoding='utf-8'))
plan['version']=2
plan['source_sha256']={}
for name in ['group_teacher_runtime_v1.py','check_group_teacher_v1.py']:
 source=w/name
 ast.parse(source.read_text(encoding='utf-8'))
 assert not (root/name).exists()
 shutil.copyfile(source,root/name)
 plan['source_sha256'][name]=sha(root/name)
plan['asset_sha256']={}
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('asset_manifest.json'))
 for item in manifest['files']:
  name=item['archive']
  include=(name.startswith('assets/CaReFlow/') and name.endswith('.py')) or name in ['assets/run_careflow.py','assets/run_control_baseline.py','assets/mosi.pkl','frozen_v5/encoder_adapter.py'] or name.startswith('assets/deberta-v3-base/')
  if not include:continue
  with z.open(name) as stream:actual=digest_stream(stream)
  assert actual==item['sha256'],name
  plan['asset_sha256'][name]=actual
plan['dataset_sha256']=plan['asset_sha256']['assets/mosi.pkl']
assert plan['dataset_sha256']=='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b'
plan['frozen_source_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
plan['implementation_stage']='Local byte-frozen mechanism precheck source; not deployed, not GPU-executed, not formal-training-authorized by mechanism evidence yet'
plan['transport_dependency']='Previous read-only remote checks rejected by approval policy. No alternate transport used to bypass the rejection.'
target=root/'group_teacher_plan_v2.json'
assert not target.exists()
target.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'plan_sha256':sha(target),'sources':plan['source_sha256'],'asset_count':len(plan['asset_sha256']),'status':plan['implementation_stage']}))

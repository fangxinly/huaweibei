from pathlib import Path
import datetime,hashlib,json,subprocess,zipfile,shutil
old=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen')
teacher=Path('D:/CodexBackups/selective_flow_20261003_1105/counterfactual_v5_completed_20261005/a/run')
protocol=json.loads((teacher/'protocol.json').read_text())
repo=old/'work/CaReFlow';helpers=old/'outputs/careflow_reproduction';backbone=old/'work/deberta-v3-base';data=old/'work/careflow_data/mosi.pkl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(data)==protocol['data_sha256']
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==protocol['author_commit']
files=[]
for name,digest in protocol['author_source_sha256'].items():
 p=repo/name;assert sha(p)==digest;files.append((p,'assets/CaReFlow/'+name))
for p in (repo/'.git').rglob('*'):
 if p.is_file():files.append((p,'assets/CaReFlow/'+p.relative_to(repo).as_posix()))
for name,digest in protocol['helper_source_sha256'].items():
 p=helpers/name;assert sha(p)==digest;files.append((p,'assets/'+name))
for name,digest in protocol['backbone_sha256'].items():
 p=backbone/name;assert sha(p)==digest;files.append((p,'assets/deberta-v3-base/'+name))
files.append((data,'assets/mosi.pkl'))
for name,digest in protocol['own_source_sha256'].items():
 p=Path('work/inflow_counterfactual_v5_deployment_20261005T0520Z')/name;assert sha(p)==digest;files.append((p,'frozen_v5/'+name))
for name in ['protocol.json','selection.json','results.json','predictions.npz','history.json','batch_orders.npy','shared_phase.json']:
 files.append((teacher/name,'teacher_a_metadata/'+name))
dest=Path('D:/CodexBackups/selective_flow_20261003_1105/new_p4_assets_20261005T1220Z');dest.mkdir(exist_ok=False)
assert shutil.disk_usage('D:/').free>3_000_000_000
rows=[{'archive':n,'bytes':p.stat().st_size,'sha256':sha(p)} for p,n in files]
with zipfile.ZipFile(dest/'common_assets.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p,name in files:z.write(p,name)
 z.writestr('asset_manifest.json',json.dumps({'files':rows,'data_policy':'Only TRAIN/dev requested by new scripts; original pickle container transferred unchanged. No TEST labels selected or used.'},indent=2))
r={'status':'NEW_NODE_COMMON_ASSETS_PACKAGED_PINNED_A_SOURCE_MATCH','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'zip':str(dest/'common_assets.zip'),'zip_bytes':(dest/'common_assets.zip').stat().st_size,'zip_sha256':sha(dest/'common_assets.zip'),'rows':rows,'teacher_weight_source':str(teacher/'best.pt'),'teacher_weight_sha256':json.loads((teacher/'selection.json').read_text())['checkpoint_sha256'],'teacher_weight_bytes':(teacher/'best.pt').stat().st_size,'teacher_weight_in_zip':False,'limits':'No remote transfer or TRAIN teacher collection verified by this record.'}
Path('outputs/新P4数据源码依赖打包核验.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
(dest/'package_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:r[k] for k in ['status','zip','zip_bytes','zip_sha256']},ensure_ascii=False))

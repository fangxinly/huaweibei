"""Freeze a new prefix execution only after genuine two-node and durable transport receipts."""
import ast,hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1];tag,stamp=sys.argv[1:]
root=base/'work'/('autonomous_mse16_qualified_'+tag);root.mkdir(exist_ok=False)
original=base/'work/autonomous_mse16_20261008T175549Z';ev=base/'outputs/autonomous_mse16_20261008T175549Z'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
for file in original.iterdir():
 if file.suffix in ('.py','.npy'):shutil.copyfile(file,root/file.name)
for n in ('run_prefix16_capture_v1.py','qualify_health_components_v2.py'):shutil.copyfile(base/'work'/n,root/n)
for node in ('A','B'):
 r=json.loads((ev/(node+'_native_result.json')).read_text());e=json.loads((ev/(node+'_native_exit.json')).read_text())
 assert r['helper_SHA']==sha(root/'training_health_components_v2.py') and r['source_SHA']==sha(root/'qualify_health_components_v2.py') and e['natural_exit']==0
 for suffix in ('native_result.json','native_exit.json'):shutil.copyfile(ev/(node+'_'+suffix),root/(node+'_'+suffix))
bt=json.loads((ev/'B_release_transport_result.json').read_text());assert bt['status']=='OTHER_NODE_PUBLIC_RELEASE_ORIGINAL_SHA_CRC_MEMBERS_VERIFIED'
shutil.copyfile(ev/'B_release_transport_result.json',root/'B_release_transport_result.json')
pub=json.loads((base/'outputs/remote_release_transport_test_public_metadata.json').read_text());assert not pub['draft'] and any(a['digest']=='sha256:'+bt['SHA'] for a in pub['assets'])
shutil.copyfile(base/'outputs/remote_release_transport_test_public_metadata.json',root/'public_release_metadata.json')
assert shutil.disk_usage('C:/').free>=500_000_000
plan=json.loads((original/'precheck_plan.json').read_text(encoding='utf8'))
plan.update(status='AUTONOMOUS_MSE_PREFIX16_REAL_EXECUTION_QUALIFIED',actualclock_qualification_UTC=stamp,
 storage_transport_qualified=True,health_v2_native_qualified=True,local_D_capacity_claimed=False,
 latest_human_autonomous_storage_choice_recorded=True,
 human_latest_authorization='自己一直迭代优化 一直处理 不要什么都要我给你一个命令一个命令的 自己直接干到天亮 或者每多长时间给自己一个定时 继续优化处理',
 storage_disposition='Agent chooses authorized GitHub Release + other-node CPU + C small originals, with no frozen-file deletions; not an invented explicit D-path approval',
 storage_existing_public_transport_release=pub['html_url'],small_C_free_actual=shutil.disk_usage('C:/').free,
 source_sha256={p.name:sha(p) for p in root.iterdir() if p.suffix in ('.py','.npy')},
 qualification_sha256={p.name:sha(p) for p in root.iterdir() if p.suffix=='.json'},
 minimum_save_margin_seconds=7200,continuation_runner_qualified=False)
(root/'qualified_precheck_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
with zipfile.ZipFile(root/'qualified_source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in root.iterdir():
  if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
with zipfile.ZipFile(root/'qualified_source.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
print(json.dumps({'root':str(root),'plan_SHA':sha(root/'qualified_precheck_plan.json'),'bundle_SHA':sha(root/'qualified_source.zip'),'D_capacity_claimed':False,'real_TRAIN_not_started_yet':True}))

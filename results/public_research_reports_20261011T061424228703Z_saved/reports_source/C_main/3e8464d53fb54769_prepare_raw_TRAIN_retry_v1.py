import datetime,hashlib,json,pathlib,shutil,zipfile,sys
P=pathlib.Path
EV=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z')
DC=EV.parent/'candidate_posttrain_lowC_20261009T005229Z'
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=sys.argv[1];tag=sys.argv[2]
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
state=json.loads((DC/'D_current_research_state.json').read_bytes())
r=EV/('raw_TRAIN_retry_prepared_actual_'+tag);r.mkdir();bundle=r/'bundle';bundle.mkdir()
sources=P(state['latest_raw_TRAIN_driver_qualification']['root'])
for n,h in state['latest_raw_TRAIN_driver_qualification']['qualification']['source_SHA'].items():
 assert sha(sources/n)==h;shutil.copyfile(sources/n,bundle/n)
pr=P(state['latest_raw_TRAIN_probe_protocol_prepared']['root'])
for n in ['frozen_raw_TRAIN_probe_protocol.json','TRAIN_video_fivefold_rows.json']:shutil.copyfile(pr/n,bundle/n)
rr=P(state['latest_raw_TRAIN_reporter_qualification']['root'])
assert sha(rr/'TRAIN_raw_probe_report_v1.py')==state['latest_raw_TRAIN_reporter_qualification']['summary']['source_SHA']
shutil.copyfile(rr/'TRAIN_raw_probe_report_v1.py',bundle/'TRAIN_raw_probe_report_v1.py')
shutil.copyfile(P(__file__).with_name('raw_TRAIN_capture_v1.py'),bundle/'raw_TRAIN_capture_v1.py')
remote='/data/coding/N2_raw_TRAIN_probe_actual_'+tag
runtime='/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z'
source_SHA={p.name:sha(p) for p in bundle.glob('*.py')}
plan=dict(actualclock_preparation_UTC=stamp,execution_identity='BATCH5_N2_RAW_TRAIN_FIXED_RIDGE',UUID='GPU-5c35273d-8d80-5183-be46-0c9107d57373',human_conservative_deadline='2026-10-10T16:30:58+00:00',interpreter=runtime+'/.venv/bin/python',data=runtime+'/assets/mosi.pkl',root=remote+'/prediction',once_token=remote+'/once_raw_TRAIN_prediction.json',protocol=remote+'/payload/frozen_raw_TRAIN_probe_protocol.json',protocol_SHA=sha(bundle/'frozen_raw_TRAIN_probe_protocol.json'),fold_manifest=remote+'/payload/TRAIN_video_fivefold_rows.json',fold_SHA=sha(bundle/'TRAIN_video_fivefold_rows.json'),source_SHA=source_SHA,no_AB_reexecution=True,no_metrics_before_prediction_preservation=True)
(bundle/'execution_plan.json').write_bytes(raw(plan))
manifest=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(bundle.iterdir())]
with zipfile.ZipFile(r/'payload.zip','x',zipfile.ZIP_DEFLATED) as z:
 for m in manifest:z.write(bundle/m['name'],m['name'])
 z.writestr('payload_member_manifest.json',raw(manifest))
with zipfile.ZipFile(r/'payload.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={m['name'] for m in manifest}|{'payload_member_manifest.json'}
 for m in manifest:assert hashlib.sha256(z.read(m['name'])).hexdigest()==m['sha256']
ref=dict(root=str(r),remote_root=remote,payload_SHA=sha(r/'payload.zip'),plan_SHA=sha(bundle/'execution_plan.json'),interpreter=plan['interpreter'],prepared_only=True,actualclock_UTC=stamp)
(r/'prepared_reference.json').write_bytes(raw(ref));shutil.copyfile(__file__,r/P(__file__).name)
print(json.dumps(ref))

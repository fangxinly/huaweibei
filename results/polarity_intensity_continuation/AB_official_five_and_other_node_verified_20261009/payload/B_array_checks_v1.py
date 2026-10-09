"""Explicit new N3 byte/metric and N2 original author-score contracts; no model execution."""
import argparse,datetime,hashlib,importlib.util,json,os,pathlib,pickle,platform,shutil,subprocess,sys,zipfile
import numpy as np
P=pathlib.Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def read(p):return json.loads(P(p).read_bytes())
def write(p,v):P(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def unzip(zp,root,expected):
 assert sha(zp)==expected
 with zipfile.ZipFile(zp) as z:
  rows=json.loads(z.read('member_manifest.json'));names=z.namelist();assert len(names)==len(set(names)) and set(names)=={v['name'] for v in rows}|{'member_manifest.json'} and z.testzip() is None
  for v in rows:
   b=z.read(v['name']);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];p=root/v['name'];assert p.resolve().is_relative_to(root.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 return len(rows)
def guard(payload,p,stage,root):
 assert sha(__file__)==p['driver_SHA'] and np.__version__=='1.26.4' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for n,h in p['payload_SHA'].items():assert sha(payload/n)==h,n
 node='N2' if stage=='score' else 'N3';uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
 import importlib.metadata as md
 runtime={n:md.version(n) for n in (p['runtime_versions'] if node=='N2' else ['numpy'])}
 if node=='N2':assert runtime==p['runtime_versions']
 assert uuid==p['nodes'][node]['UUID'] and not compute.strip()
 remaining=(datetime.datetime.fromisoformat(p['nodes'][node]['deadline'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>7800
 mem=int(next(v.split()[1] for v in P('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024;assert mem>=6*1024**3 and shutil.disk_usage(root.parent).free>1024**3
 facts=dict(actual_UTC=utc(),node=node,UUID=uuid,hostname=platform.node(),fullargv=[sys.executable]+sys.argv,python=sys.executable,numpy=np.__version__,available_RAM=mem,free_bytes=shutil.disk_usage(root.parent).free,remaining_seconds=remaining,empty_compute=compute,process_fullargv=subprocess.check_output(['ps','-eo','pid,ppid,stat,comm,args'],text=True),source_SHA=sha(__file__),plan_SHA=sha(payload/'plan.json'))
 facts['runtime_versions']=runtime;write(root/'fresh_guard.json',facts);return facts
def work(payload,p,stage,root):
 fresh=guard(payload,p,stage,root);m=module('original_independent',payload/'independent_original.py');metric=module('original_author_metric',payload/'sentiment_metrics_careflow_v1.py');write(root/'synthetic_qualification.json',m.synthetic(metric))
 orig=root/'prediction_original';count=unzip(payload/'prediction_complete.zip',orig,p['prediction_archive_SHA']);r=read(orig/'out/inference_result.json');npz=orig/'out/fixed_official_VAL_TEST_prediction.npz';assert sha(npz)==r['prediction_SHA']==p['prediction_SHA'] and r['selected_state_SHA']==p['selected_state_SHA'];assert read(orig/'natural_exit.json')['natural_exit']==0 and r['scalar_VAL_TEST_true_labels_not_indexed'] and r['all_parameters_buffers_RNG_unchanged']
 pred=dict(np.load(npz,allow_pickle=False));arrays={}
 for role,n in [('VAL',229),('TEST',685)]:
  ids=pred[role+'_ids'];on=pred[role+'_prediction'];off=pred[role+'_p0'];assert ids.tolist()==p['official_role_IDs'][role] and len(ids)==len(set(ids.tolist()))==n and on.shape==off.shape==(n,) and np.isfinite(on).all() and np.isfinite(off).all()
 for n,a in pred.items():
  arrays[n]=dict(shape=list(a.shape),dtype=str(a.dtype),byte_SHA=hashlib.sha256(a.tobytes()).hexdigest());assert arrays[n]==p['array_manifest'][n]
 if stage=='byte':
  assert platform.node()!=p['inference_host'];write(root/'result.json',dict(status='B_N3_OTHER_NODE_ORIGINAL_PREDICTION_BYTES_PASSED',actual_UTC=utc(),node='N3',hostname=platform.node(),UUID=fresh['UUID'],prediction_SHA=sha(npz),archive_SHA=p['prediction_archive_SHA'],member_count=count,selected_state_SHA=p['selected_state_SHA'],arrays=arrays,official_IDs_verified=True,true_labels_not_consumed=True,all_member_SHA_CRC_unique_passed=True,Release_original_digest_verified=p['prediction_publication']['remote_digest_verified'],different_node_from_inference=True));return
 assert sha(payload/'mosi.pkl')==p['dataset_SHA']
 if stage=='score':
  gate=read(payload/'byte_gate.json');assert gate['status']=='B_N3_OTHER_NODE_ORIGINAL_PREDICTION_BYTES_PASSED' and gate['prediction_SHA']==p['prediction_SHA'] and gate['selected_state_SHA']==p['selected_state_SHA']
  sp=dict(p['author_protocol']);sp.update(prediction_SHA=p['prediction_SHA'],prediction_Release_B_CPU_gate=True,score_once_token=p['author_once_token']);sp['source_sha256']=p['original_source_SHA'];sp['asset_sha256']=p['original_asset_SHA'];write(root/'original_author_protocol.json',sp)
  argv=[sys.executable,'-B',p['original_bundle']+'/score_official.py','--plan',str(root/'original_author_protocol.json'),'--plan-sha',sha(root/'original_author_protocol.json'),'--bundle',p['original_bundle'],'--assets',p['original_assets'],'--prediction',str(npz),'--out',str(root/'out')]
  write(root/'author_dispatch.json',dict(actual_UTC=utc(),fullargv=argv));code=subprocess.call(argv,stdin=subprocess.DEVNULL);write(root/'author_natural_exit.json',dict(actual_UTC=utc(),natural_exit=code));assert code==0
  result=read(root/'out/official_aligned_five_result.json');assert result['selected_state_SHA']==p['selected_state_SHA'] and result['prediction_SHA']==p['prediction_SHA'];write(root/'result.json',dict(status='B_N2_ORIGINAL_AUTHOR_FIVE_ONCE_COMPLETE',actual_UTC=utc(),hostname=platform.node(),node='N2',selected_epoch=92,selected_state_SHA=p['selected_state_SHA'],prediction_SHA=p['prediction_SHA'],official_dataset_SHA=p['dataset_SHA'],author_result_SHA=sha(root/'out/official_aligned_five_result.json'),roles=result['roles'],primary_A_scoring_not_repeated=True,other_node_metric_pending=True));return
 author=read(payload/'author_result.json');assert sha(payload/'author_result.json')==p['author_result_SHA'] and author['prediction_SHA']==p['prediction_SHA'] and platform.node()!=p['author_host']
 with (payload/'mosi.pkl').open('rb') as f:data=pickle.load(f)
 roles={};maximum=0.
 for role,key in [('VAL','dev'),('TEST','test')]:
  ids=[v[2].decode() if isinstance(v[2],bytes) else v[2] for v in data[key]];assert ids==p['official_role_IDs'][role];y=np.asarray([np.asarray(v[1]).reshape(-1)[0] for v in data[key]],np.float32).astype(float);roles[role]={}
  for mode,suffix in [('new','_prediction'),('messages_off','_p0')]:
   table=m.independent(pred[role+suffix],y);roles[role][mode]=table
   for k in m.KEYS:
    e=abs(table[k]-author['roles'][role][mode][k]);maximum=max(maximum,e);assert e<1e-12
 write(root/'result.json',dict(status='B_N3_OTHER_NODE_OFFICIAL_ORIGINAL_NPZ_ALL_FIVE_VERIFIED',actual_UTC=utc(),node='N3',hostname=platform.node(),UUID=fresh['UUID'],author_host=p['author_host'],different_host_actual=True,selected_epoch=92,selected_state_SHA=p['selected_state_SHA'],prediction_SHA=p['prediction_SHA'],official_dataset_SHA=p['dataset_SHA'],author_result_SHA=p['author_result_SHA'],roles=roles,max_error=maximum,primary_author_scoring_not_repeated=True))
def capture(payload,stage,root):
 root.mkdir();p=read(payload/'plan.json');write(root/'dispatch.json',dict(actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv));code=0
 try:work(payload,p,stage,root)
 except Exception:
  import traceback;code=1;(root/'failure.log').write_text(traceback.format_exc())
 write(root/'natural_exit.json',dict(actual_UTC=utc(),pid=os.getpid(),natural_exit=code));s=root/'source';s.mkdir()
 for f in payload.iterdir():
  if f.suffix in ('.py','.json'):shutil.copyfile(f,s/f.name)
 files=[f for f in sorted(root.rglob('*')) if f.is_file()];rows=[dict(name=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files];zpath=root/'complete_actual_original.zip'
 with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
  for f,row in zip(files,rows):z.write(f,row['name'])
  z.writestr('member_manifest.json',json.dumps(rows,indent=2))
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for row in rows:assert hashlib.sha256(z.read(row['name'])).hexdigest()==row['sha256']
 write(root/'capture_receipt.json',dict(actual_UTC=utc(),stage=stage,natural_exit=code,archive=str(zpath),archive_SHA=sha(zpath),archive_bytes=zpath.stat().st_size,all_member_SHA_CRC_unique_passed=True));return code
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--payload',type=P,required=True);a.add_argument('--stage',choices=['byte','score','peer'],required=True);a.add_argument('--root',type=P,required=True);v=a.parse_args();sys.exit(capture(v.payload,v.stage,v.root))

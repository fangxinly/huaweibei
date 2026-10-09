"""Verify actual original capture or publish one exact small artifact."""
import argparse,datetime,hashlib,json,pathlib,zipfile,os,sys,urllib.request,urllib.parse,urllib.error
P=pathlib.Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def checkzip(path,manifestname):
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  manifest=json.loads(z.read(manifestname));assert set(z.namelist())=={v['name'] for v in manifest}|{manifestname}
  for v in manifest:
   assert not P(v['name']).is_absolute() and '..' not in P(v['name']).parts
   b=z.read(v['name']);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 return manifest
def verify(root):
 o=root/'remote_original';receipt=json.loads((o/'wrapper_capture_receipt.json').read_bytes());capture=o/'complete_actual_raw_TRAIN_capture.zip'
 assert sha(capture)==receipt['archive_SHA'] and receipt['natural_exit']==0
 members=checkzip(capture,'capture_member_manifest.json');dest=root/'verified_original';assert not dest.exists()
 with zipfile.ZipFile(capture) as z:z.extractall(dest)
 exitrec=json.loads((dest/'natural_exit.json').read_bytes());assert exitrec['natural_exit']==0
 inner=dest/'prediction/complete_raw_TRAIN_prediction_original.zip';proof=json.loads((dest/'prediction/capture_receipt.json').read_bytes());assert sha(inner)==proof['sha256']
 checkzip(inner,'member_manifest.json')
 plan=json.loads((dest/'payload/execution_plan.json').read_bytes())
 for name,h in plan['source_SHA'].items():assert sha(dest/'payload'/name)==h
 result=json.loads((dest/'prediction/prediction_result.json').read_bytes())
 assert not result['VAL_TEST_numeric_decode'] and result['TRAIN_rows']==1281 and result['video_groups']==52 and result['official_AB_tokens_untouched']
 assert sha(dest/'prediction/TRAIN_fixed_fivefold_prediction.npz')==result['prediction_SHA']
 assert sha(dest/'prediction/fit_fold_states.npz')==result['states_SHA']
 proof=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(capture),archive_SHA=sha(capture),archive_bytes=capture.stat().st_size,natural_exit=0,all_member_SHA_CRC_unique_exact_set_passed=True,inner_prediction_archive_SHA=sha(inner),prediction_SHA=result['prediction_SHA'],plan_SHA=sha(dest/'payload/execution_plan.json'),metrics_not_computed=True,content_members=len(members))
 (root/'prediction_D_preservation_receipt.json').write_bytes(raw(proof));print(json.dumps(proof))
def publish(path,expected,name,receipt):
 assert sha(path)==expected and path.stat().st_size<10*1024**2
 sys.path.insert(0,str(P(__file__).resolve().parent));import publish_release_assets_v1 as publisher
 os.environ['PATH']='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH']
 state=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(path),source_SHA=expected,status='PUBLICATION_STARTED',name=name)
 receipt.write_bytes(raw(state))
 try:
  headers=publisher.credential_headers();release=publisher.request('https://api.github.com/repos/fangxinly/huaweibei/releases/tags/autonomous-flow-health-20261008',headers)
  found=None;page=1
  while True:
   assets=publisher.request(release['assets_url']+'?per_page=100&page='+str(page),headers)
   found=next((a for a in assets if a['name']==name),None)
   if found is not None or len(assets)<100:break
   page+=1
  if found is None:
   payload=path.read_bytes();req=urllib.request.Request(release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(name),data=payload,headers=dict(headers,**{'Content-Type':'application/octet-stream','Content-Length':str(len(payload))}),method='POST')
   with urllib.request.urlopen(req,timeout=45) as response:found=json.load(response)
  assert found['state']=='uploaded' and found['size']==path.stat().st_size and found.get('digest')=='sha256:'+expected
  state.update(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PUBLICATION_REMOTE_DIGEST_VERIFIED',id=found['id'],url=found['browser_download_url'],remote_digest_verified=True)
  receipt.write_bytes(raw(state));print(json.dumps(state))
 except Exception as exc:
  state.update(status='PUBLICATION_FAILED_ORIGINAL_PRESERVED',error_type=type(exc).__name__)
  if isinstance(exc,urllib.error.HTTPError):state['HTTP_status']=exc.code
  receipt.write_bytes(raw(state));raise RuntimeError(state['status']) from None
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify',type=P);p.add_argument('--publish',type=P);p.add_argument('--sha');p.add_argument('--name');p.add_argument('--receipt',type=P);a=p.parse_args()
 if a.verify:assert not a.publish;verify(a.verify)
 else:assert a.publish and a.sha and a.name and a.receipt;publish(a.publish,a.sha,a.name,a.receipt)

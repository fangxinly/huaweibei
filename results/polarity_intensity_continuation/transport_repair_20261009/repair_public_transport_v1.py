"""Repair short HTTP downloads; reuse the unchanged qualified CPU audit."""
import argparse, datetime, hashlib, importlib, importlib.metadata, io, json, os
import pathlib, re, shutil, subprocess, sys, time, urllib.request, zipfile
P=pathlib.Path
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()
def download(row,dest,log,opener=urllib.request.urlopen,chunk=64*1024**2):
 size=row['bytes']; errors=0
 assert not dest.exists()
 dest.touch()
 while dest.stat().st_size<size:
  start=dest.stat().st_size; end=min(size-1,start+chunk-1)
  req=urllib.request.Request(row['url'],headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
  try:
   with opener(req,timeout=120) as response:
    status=response.status
    if status==206:
     m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
     assert m and int(m[1])==start and int(m[2])<=end and int(m[3])==size
     wanted=int(m[2])-start+1
    else:
     assert status==200 and start==0, ('Range was ignored',status,start)
     wanted=end-start+1
    got=0
    with dest.open('ab') as out:
     while got<wanted:
      block=response.read(min(8*1024**2,wanted-got))
      if not block: break
      out.write(block); got+=len(block)
    assert got>0, 'Empty HTTP response'
   errors=0
   with log.open('a') as f: f.write(json.dumps({'actual_UTC':utc(),'name':row['name'],'start':start,'requested_end':end,'response_status':status,'received':got,'committed':dest.stat().st_size})+'\n')
  except (OSError,TimeoutError) as e:
   errors+=1
   with log.open('a') as f: f.write(json.dumps({'actual_UTC':utc(),'name':row['name'],'start':start,'error_type':type(e).__name__,'consecutive_errors':errors})+'\n')
   if errors>=3: raise
   time.sleep(2)
 assert dest.stat().st_size==size and digest(dest)==row['sha256'], row['name']
def qualify(root):
 root.mkdir(); data=bytes(range(251))*39; calls=[]
 class Response(io.BytesIO):
  def __init__(self,start,end,short):
   super().__init__(data[start:min(end+1,start+short)])
   self.status=206; self.headers={'Content-Range':f'bytes {start}-{end}/{len(data)}'}
 def fake(req,timeout):
  start,end=map(int,req.get_header('Range')[6:].split('-')); calls.append((start,end))
  return Response(start,end,137 if len(calls)==1 else end-start+1)
 row={'name':'synthetic','url':'https://fixture.invalid','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 download(row,root/'actual.bin',root/'transport.jsonl',fake,chunk=512)
 assert (root/'actual.bin').read_bytes()==data and calls[1][0]==137
 bad=dict(row,sha256='0'*64)
 try: download(bad,root/'bad.bin',root/'bad.jsonl',fake,chunk=512)
 except AssertionError: pass
 else: raise AssertionError('Corrupt digest accepted')
 write(root/'qualification_result.json',{'actual_UTC':utc(),'status':'SHORT_HTTP_RESUME_AND_BAD_SHA_REJECTION_PASSED','source_SHA':digest(P(__file__)),'real_data_or_labels_indexed':False,'real_network':False})
def child(node,root):
 base=P('/data/coding'); bundle=base/(node+'_audit_C_qualified_20261009T040110Z')
 runtime=base/'C_candidate_pureCPU_runtime_actual_20261009T004259Z'
 plan=bundle/'official_audit_protocol.json'; p=json.loads(plan.read_text())
 expected={'A':'16db6344a0bb883dc797c3e5b430315b3cb8912326229d7e999293af49c1926a','B':'11a1a8841ff942795100d09e54f575c53f1ec827c97fc70eca803bc49a57a8b5'}[node]
 assert digest(plan)==expected and sys.executable==str(runtime/'.venv/bin/python')
 sys.path.insert(0,str(bundle)); from common import verify
 verify(p,bundle,runtime)
 assert {n:importlib.metadata.version(n) for n in p['runtime_versions']}==p['runtime_versions']
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 assert uuid==p['GPU_UUID']['C'] and not compute.strip()
 remaining=(datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
 assert remaining>14400 and shutil.disk_usage(base).free>14_000_000_000
 write(root/'fresh_preflight.json',{'actual_UTC':utc(),'UUID':uuid,'compute':compute,'runtime_versions':p['runtime_versions'],'processes':subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),'space_free':shutil.disk_usage(base).free,'remaining_seconds':remaining,'source_SHA':digest(P(__file__)),'plan_SHA':expected,'fullargv':[sys.executable]+sys.argv})
 publication=base/(node+'100_publication_receipt_20261009T040110Z.json'); cap=base/(node+'100_capture_receipt_20261009T040110Z.json')
 pub=json.loads(publication.read_text()); c=json.loads(cap.read_text())
 assert pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and c['natural_exit']==0
 rows=sorted([r for r in pub['assets'] if r['source_sha256']==c['archive_SHA']],key=lambda r:r['source_offset'])
 parts=root/'verified_public_parts'; parts.mkdir(); local={}
 for row in rows:
  dest=parts/row['name']; download(row,dest,root/'transport.jsonl'); local[row['url']]=dest
 write(root/'verified_public_parts.json',{'actual_UTC':utc(),'parts':[{**r,'local_SHA':digest(local[r['url']])} for r in rows],'qualified_audit_source_unchanged':True})
 # Supply complete SHA-verified public response bytes to the unchanged qualified
 # downloader. All archive/member, state, Adam, RNG and selection checks remain.
 original=urllib.request.urlopen
 def verified_open(url,timeout=180):
  assert isinstance(url,str) and url in local
  return local[url].open('rb')
 urllib.request.urlopen=verified_open
 try:
  mod=importlib.import_module('download_and_audit_final_v1')
  from types import SimpleNamespace
  mod.run(SimpleNamespace(plan=plan,plan_sha=expected,bundle=bundle,assets=runtime,out=root/'out',publication=publication,capture=cap))
 finally: urllib.request.urlopen=original
def pair(root):
 root.mkdir(); shutil.copy2(__file__,root/'repair_public_transport_v1.py')
 runtime=P('/data/coding/C_candidate_pureCPU_runtime_actual_20261009T004259Z/.venv/bin/python')
 results=[]
 for node in ['A','B']:
  nr=root/node; nr.mkdir(); cmd=[str(runtime),str(root/'repair_public_transport_v1.py'),'--child',node,'--root',str(nr)]
  with (nr/'stdout.log').open('wb') as out,(nr/'stderr.log').open('wb') as err:
   proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
   write(nr/'dispatch.json',{'actual_UTC':utc(),'pid':proc.pid,'fullargv':cmd}); code=proc.wait()
  write(nr/'natural_exit.json',{'actual_UTC':utc(),'pid':proc.pid,'fullargv':cmd,'natural_exit':code})
  files=[f for f in nr.rglob('*') if f.is_file() and (f.suffix in ['.json','.jsonl','.log']) and f.stat().st_size<40_000_000]
  members={f.relative_to(nr).as_posix():digest(f) for f in files}; write(nr/'member_SHA.json',members)
  archive=nr/'complete_small_original.zip'
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
   for name in list(members)+['member_SHA.json']: z.write(nr/name,name)
   z.write(root/'repair_public_transport_v1.py','original_source/repair_public_transport_v1.py')
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
   for name,h in members.items(): assert hashlib.sha256(z.read(name)).hexdigest()==h
  assert archive.stat().st_size<=40_000_000
  rec={'actual_UTC':utc(),'natural_exit':code,'archive_SHA':digest(archive),'archive_bytes':archive.stat().st_size,'ZIP_CRC_unique_all_members':True,'original_failed_roots_untouched':True,'large_originals_retained':True}
  write(nr/'capture_receipt.json',rec); results.append({'node':node,**rec})
 write(root/'pair_natural_complete.json',{'actual_UTC':utc(),'results':results})
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--root',type=P,required=True); p.add_argument('--qualify',action='store_true'); p.add_argument('--child',choices=['A','B']); a=p.parse_args()
 if a.qualify: qualify(a.root)
 elif a.child: child(a.child,a.root)
 else: pair(a.root)

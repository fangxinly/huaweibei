import pathlib,json,hashlib,zipfile,subprocess,os,datetime,shutil
base=pathlib.Path('/data/coding')
clock=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
root=base/'C_pair_full4000_coordinator_actual_20261009T040237Z'
root.mkdir()
source_sha=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert uuid=='GPU-daa2c09a-4ce5-26dd-b375-6b61242c6795'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(base).free>15_000_000_000
runtime=base/'C_candidate_pureCPU_runtime_actual_20261009T004259Z'
items=[('A','3072f8e1f3b212d806185029f11a7512d687cf8b4a526e9a50f901806cae273d','16db6344a0bb883dc797c3e5b430315b3cb8912326229d7e999293af49c1926a'),('B','517858701d2eefa5f596dd6d0c4ab058e4f0e8d34492ad8919d81d44d0a08dfa','11a1a8841ff942795100d09e54f575c53f1ec827c97fc70eca803bc49a57a8b5')]
rows=[]
for node,zsha,psha in items:
 zpath=base/(node+'_audit_C_stage_source_20261009T040110Z.zip')
 assert hashlib.sha256(zpath.read_bytes()).hexdigest()==zsha
 bundle=base/(node+'_audit_C_qualified_20261009T040110Z')
 bundle.mkdir()
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n in z.namelist():assert (bundle/n).resolve().is_relative_to(bundle.resolve())
  z.extractall(bundle)
 plan=bundle/'official_audit_protocol.json'
 assert hashlib.sha256(plan.read_bytes()).hexdigest()==psha
 cmd=[str(runtime/'.venv/bin/python'),str(bundle/'run_posttrain_capture_v1.py'),'--stage','audit','--node','C','--plan',str(plan),'--plan-sha',psha,'--assets',str(runtime),'--root',str(base/('C_'+node+'_candidate_full4000_audit_actual_20261009T040237Z')),'--publication',str(base/(node+'100_publication_receipt_20261009T040110Z.json')),'--capture',str(base/(node+'100_capture_receipt_20261009T040110Z.json'))]
 rows.append({'node':node,'cmd':cmd})
(root/'coordinator_start.json').write_text(json.dumps({'actual_UTC':clock(),'UUID':uuid,'source_SHA':source_sha,'fullargv':[str(pathlib.Path(__file__))],'sequential_CPU_only':True,'no_training_rerun':True,'commands':rows},indent=2))
for row in rows:
 node=row['node']
 with (root/(node+'_wrapper_stdout.log')).open('wb') as out,(root/(node+'_wrapper_stderr.log')).open('wb') as err:
  p=subprocess.Popen(row['cmd'],stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
  (root/(node+'_wrapper_dispatch.json')).write_text(json.dumps({'actual_UTC':clock(),'pid':p.pid,'fullargv':row['cmd']},indent=2))
  code=p.wait()
 (root/(node+'_wrapper_natural_exit.json')).write_text(json.dumps({'actual_UTC':clock(),'pid':p.pid,'fullargv':row['cmd'],'natural_exit':code},indent=2))
(root/'coordinator_natural_complete.json').write_text(json.dumps({'actual_UTC':clock(),'all_wrappers_observed':True},indent=2))

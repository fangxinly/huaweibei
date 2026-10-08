"""Independent stdlib audit of captured failed precheck, never model forward."""
import argparse,datetime,hashlib,json,math,pathlib,zipfile
p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--receipt',required=True);p.add_argument('--capture-exit',required=True);p.add_argument('--initial',required=True);p.add_argument('--initial-cpu-receipt',required=True);p.add_argument('--out',required=True);a=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
r=read(a.receipt);ex=read(a.capture_exit);cp=read(a.initial_cpu_receipt)
assert sha(a.snapshot)==r['sha256'] and pathlib.Path(a.snapshot).stat().st_size==r['bytes']
assert ex['capture_actual_exit_code']==0 and ex['natural_wait_verified']
with zipfile.ZipFile(a.snapshot) as z:
 names=z.namelist();assert z.testzip() is None and len(names)==len(set(names))==r['members']
 m=json.loads(z.read('member_manifest.json'));assert set(names)==set(m)|{'member_manifest.json'}
 for n,x in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==x['sha256'] and len(z.read(n))==x['bytes']
 inv=json.loads(z.read('inventory.json'));natural=json.loads(z.read('run/natural_exit.json'));outer=json.loads(z.read('run/actual_wrapper_exit.json'))
 assert inv['gpu_uuid']=='GPU-53696803-875e-eec8-2231-29db63579891' and not inv['compute']
 assert natural['exit_code']==outer['wrapper_exit_code']==1 and natural['natural_exit'] and not natural['actual_scientific_precheck_complete']
 assert natural['child_pid']==inv['launch']['child_pid']
 assert hashlib.sha256(z.read('run/child_stderr.log')).hexdigest()==natural['raw_stderr_sha256']
 assert b'ACTUAL_PRECHECK_TIME_OR_GPU_BUDGET_EXCEEDED: mixed_singleton' in z.read('run/child_stderr.log')
 assert r['capture_source_sha256']==ex['capture_source_sha256']==hashlib.sha256(z.read('source/capture_new_fixed_precheck_v21.py')).hexdigest()
 ep=json.loads(z.read('source/deployment_execution_plan_v2.json'))
 for n,h in {**ep['source_sha256'],**ep['role_and_order_sha256']}.items():assert hashlib.sha256(z.read('source/'+n)).hexdigest()==h
 rt=json.loads(z.read('source/runtime_candidate_plan.json'));large=json.loads(z.read('large_file_manifest.json'))
 for n,h in rt['asset_sha256'].items():assert (large['public_assets/'+n]['sha256'] if 'public_assets/'+n in large else hashlib.sha256(z.read('public_assets/'+n)).hexdigest())==h
 progress=json.loads(z.read('run/out/partial_actual_progress.json'));trace=[json.loads(s) for s in z.read('run/out/actual_budget_trace.log').decode().splitlines()]
 assert progress['optimizer_steps_completed']==2 and not progress['full_INNER_replay_completed']
 assert [x['stage'] for x in trace]==['construction','initial_donor_mechanism','normal','mixed_singleton']
 assert all(t['pid']==natural['child_pid'] and t['budget_Torch_bytes']==6*1024**3 and t['budget_seconds']==1200 for t in trace)
 assert trace[-1]['elapsed_seconds']<1200 and trace[-1]['peak_allocated_bytes']<6*1024**3<trace[-1]['peak_reserved_bytes']
 assert all(a['peak_allocated_bytes']<=b['peak_allocated_bytes'] and a['peak_reserved_bytes']<=b['peak_reserved_bytes'] for a,b in zip(trace,trace[1:]))
 grads=progress['normal_and_mixed_gradients'];assert len(grads)==2
 for g in grads:
  assert len(g)==progress['construction']['parameter_tensors']
  assert all(math.isfinite(x['l1']) and x['l1']>=0 and x['elements']>0 for x in g.values())
  assert sum(x['elements'] for x in g.values())==progress['construction']['parameters']
 for label,step in zip(('normal','mixed_singleton'),progress['step_times']):
  assert step['batch_rows']==rt['precheck_row_candidates'][label] and step['seconds']>0 and math.isfinite(step['objective'])
 assert progress['step_times'][1]['actual_content_lengths'][:3]==[1,1,1]
 for c in progress['component_gradients']:
  assert c['primary']['backward']['none'] and not c['auxiliary']['backward']['none'] and not c['primary']['decoder']['none'] and c['auxiliary']['decoder']['none']
 initial=large['run/out/clean_initial_full.pt'];assert sha(a.initial)==initial['sha256']==cp['checkpoint_file_sha256'] and pathlib.Path(a.initial).stat().st_size==initial['bytes']==cp['checkpoint_bytes']
 assert cp['metadata']==progress['initial_full']['metadata'] and cp['metadata']['optimizer_steps']==0 and cp['complete_state_tensors']==374 and cp['public_encoder_exact_matched_tensors']==198
 assert 'run/out/after_two_steps_full.pt' not in large and 'run/out/actual_precheck_receipt.json' not in names
 result={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FAILED_V4_RAW_CAPTURE_AND_INITIAL_FULL_CPU_JOIN_PASSED','auditor_source_sha256':sha(__file__),'snapshot_sha256':r['sha256'],'snapshot_members':len(names),'capture_actual_utc':r['actual_utc'],'natural_child_pid':natural['child_pid'],'natural_child_exit_code':1,'stage_trace':trace,'partial_actual_optimizer_steps':2,'partial_all_retained_gradients':len(grads[0]),'step_times':progress['step_times'],'initial_full_sha256':initial['sha256'],'initial_is_existing_preserved_full_original_not_new_download':True,'initial_CPU_receipt_sha256':sha(a.initial_cpu_receipt),'public_encoder_exact_tensors':198,'full_INNER_replay_pass':False,'GPU_precheck_pass':False,'CPU_model_forward':False,'Torch_imported':False,'GPU_used':False,'formal100_started':False}
 pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('stage_trace','step_times')}))

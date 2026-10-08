"""Independent complete captured GPU precheck and D full-file audit; no Torch."""
import datetime,hashlib,io,json,math,pathlib,zipfile
import numpy as np
base=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_activation_checkpoint_actual_20261006T144615Z');d=base/'a'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
receipt=read(d/'capture_receipt.json');ex=read(d/'actual_capture23_exit.json')
assert sha(d/'snapshot.zip')==receipt['sha256'] and ex['capture_actual_exit_code']==0 and ex['natural_wait_verified']
with zipfile.ZipFile(d/'snapshot.zip') as z:
 names=z.namelist();assert z.testzip() is None and len(names)==len(set(names))==receipt['members']==98
 members=json.loads(z.read('member_manifest.json'));assert set(names)==set(members)|{'member_manifest.json'}
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256'] and len(z.read(n))==m['bytes']
 inv=json.loads(z.read('inventory.json'));large=json.loads(z.read('large_file_manifest.json'));natural=json.loads(z.read('run/natural_exit.json'));outer=json.loads(z.read('run/actual_wrapper_exit.json'));r=json.loads(z.read('run/out/actual_precheck_receipt.json'))
 assert inv['gpu_uuid']=='GPU-53696803-875e-eec8-2231-29db63579891' and not inv['compute']
 assert natural['exit_code']==outer['wrapper_exit_code']==0 and natural['natural_exit'] and natural['actual_scientific_precheck_complete'] and natural['child_pid']==r['pid']==960
 assert hashlib.sha256(z.read('run/child_stderr.log')).hexdigest()==natural['raw_stderr_sha256']
 assert r['optimizer_steps']==2 and r['tail_optimizer_steps']==0 and r['inner_rows']==153 and not r['inner_labels_read'] and r['inner_dummy_label_replacement_max_error']==r['strict_full_disk_replay_max_error']==0
 assert receipt['capture_source_sha256']==ex['capture_source_sha256']==hashlib.sha256(z.read('source/capture_new_fixed_precheck_v23.py')).hexdigest()
 ep=json.loads(z.read('source/deployment_execution_plan_v4.json'));rt=json.loads(z.read('source/runtime_candidate_plan.json'))
 for n,h in {**ep['source_sha256'],**ep['role_and_order_sha256']}.items():assert hashlib.sha256(z.read('source/'+n)).hexdigest()==h
 for n,h in rt['asset_sha256'].items():assert (large['public_assets/'+n]['sha256'] if 'public_assets/'+n in large else hashlib.sha256(z.read('public_assets/'+n)).hexdigest())==h
 trace=[json.loads(s) for s in z.read('run/out/actual_budget_trace.log').decode().splitlines()]
 assert trace[-1]['stage']=='final' and all(max(t['peak_allocated_bytes'],t['peak_reserved_bytes'])<=6*1024**3 and t['elapsed_seconds']<1200 and t['budget_Torch_bytes']==6*1024**3 and t['pid']==960 for t in trace)
 assert all(a['peak_allocated_bytes']<=b['peak_allocated_bytes'] and a['peak_reserved_bytes']<=b['peak_reserved_bytes'] for a,b in zip(trace,trace[1:]))
 comparison=[json.loads(s) for s in z.read('run/out/parent_numeric_comparison.log').decode().splitlines()]
 assert len(comparison)==2 and all(c['objective_abs_gap']==c['max_gradient_l1_norm_abs_gap']==0 for c in comparison)
 parent=json.loads(z.read('source/parent_v4_partial_actual_progress.json'));assert parent['normal_and_mixed_gradients']==r['normal_and_mixed_gradients']
 for grads in r['normal_and_mixed_gradients']+[r['tail_gradients']]:
  assert len(grads)==364 and all(math.isfinite(x['l1']) and x['l1']>=0 and x['elements']>0 for x in grads.values()) and sum(x['elements'] for x in grads.values())==r['construction']['parameters']==185402807
 for step,label in zip(r['step_times'],('normal','mixed_singleton')):assert step['batch_rows']==rt['precheck_row_candidates'][label]
 assert r['step_times'][1]['actual_content_lengths'][:3]==[1,1,1]
 assert all(j['role'] in ('fit','inner') and (not j['labels_read'] or j['role']=='fit') for j in r['original_guard_journal']+r['reloaded_guard_journal'])
 initial_path=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_precheck_failure_actual_20261006T141445Z/a/clean_initial_full.pt')
 for key,path in [('initial_full',initial_path),('after_two_steps_full',d/'after_two_steps_full.pt')]:
  pending=path.with_name(path.name+'.pending');actual=path if path.exists() else pending
  assert actual.stat().st_size==r[key]['bytes'] and sha(actual)==r[key]['file_sha256']==large['run/out/'+path.name]['sha256']
  if actual==pending:
   assert actual.resolve().parent==d.resolve() and path.resolve().parent==d.resolve();actual.rename(path)
 cp=read(initial_path.parent.parent/'B_initial_cpu_audit_receipt.json');assert cp['metadata']==r['initial_full']['metadata'] and cp['checkpoint_file_sha256']==r['initial_full']['file_sha256']
 with np.load(io.BytesIO(z.read('run/out/inner_precheck_predictions.npz')),allow_pickle=False) as a:
  assert np.array_equal(a['row_ids'],np.load(io.BytesIO(z.read('source/inner_0.npy')),allow_pickle=False)) and a['prediction'].shape==(153,) and np.isfinite(a['prediction']).all() and str(a['model_state_sha256'])==r['after_two_steps_full']['metadata']['state_sha256']
 for n,declared in zip(('donor_mechanism_clean_initial.npz','donor_mechanism_after_two_steps.npz'),r['donor_mechanism']):
  with np.load(io.BytesIO(z.read('run/out/'+n)),allow_pickle=False) as a:
   assert all(np.isfinite(a[k]).all() for k in a.files) and np.array_equal(a['0_first_state'],a['1_first_state'])
   for k in ('prediction','terminal_state','context','feedback'):assert float(np.max(np.abs(a['0_'+k]-a['2_'+k])))<=1e-6
   for label,k in [('terminal_scalar_change_max','prediction'),('second_euler_state_change_max','terminal_state'),('context_change_max','context')]:assert float(np.max(np.abs(a['0_'+k]-a['1_'+k])))==declared[label]
 # Extract small original run/source files for CPU replay-audit transport only.
 for n in names:
  if n.startswith(('run/','source/')):
   target=d/'original_small_files'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
 result={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ACTUAL_V6_GPU_PRECHECK_RAW_SOURCE_ARRAY_BUDGET_AND_D_FULL_FILE_AUDIT_PASSED','capture_actual_utc':receipt['actual_utc'],'capture_sha256':receipt['sha256'],'members':len(names),'child_pid':960,'natural_exit_code':0,'GPU_precheck_complete':True,'optimizer_steps':2,'tail_optimizer_steps':0,'parameter_tensors_all_finite_nonNone':364,'parameters':185402807,'final_budget':r['final_budget'],'numeric_comparison':comparison,'inner_rows':153,'inner_true_labels_used':False,'dummy_label_replacement_error':0.0,'strict_full_disk_replay_error':0.0,'initial_full_D_existing_original_sha256':r['initial_full']['file_sha256'],'after_two_steps_full_D_new_original_sha256':r['after_two_steps_full']['file_sha256'],'after_two_steps_full_D_bytes':r['after_two_steps_full']['bytes'],'donor_mechanism':r['donor_mechanism'],'original_other_node_complete_two_state_CPU_audit_pending':True,'formal100_started':False,'new_performance_scores':False,'Torch_imported_locally':False}
 (base/'local_joint_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('numeric_comparison','donor_mechanism')}))

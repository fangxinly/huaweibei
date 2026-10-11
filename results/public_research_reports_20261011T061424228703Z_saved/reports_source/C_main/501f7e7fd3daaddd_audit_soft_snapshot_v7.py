from pathlib import Path
import argparse,datetime,hashlib,json,zipfile,numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest();reports={};common={};histories={}
expected={'a':('fixed','GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7'),'b':('scalar','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'),'c':('soft_projected','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')}
for node,(mode,uuid) in expected.items():
 d=a.directory/node;receipt=json.loads((d/'receipt.json').read_text());b=(d/'snapshot.zip').read_bytes();assert sha(b)==receipt['sha256'] and len(b)==receipt['bytes']
 with zipfile.ZipFile(d/'snapshot.zip') as z:
  assert z.testzip() is None;manifest=json.loads(z.read('member_manifest.json'))
  assert set(z.namelist())==set(manifest)|{'member_manifest.json'}
  for n,m in manifest.items():
   data=z.read(n);assert len(data)==m['bytes'] and sha(data)==m['sha256']
  assert sha(z.read('source/capture_soft_vector_v7.py'))==json.loads(z.read('inventory.json'))['capture_source_sha256']
  get=lambda n:json.loads(z.read(n));inv=get('inventory.json');protocol=get('run/protocol.json');plan=get('source/formal_plan_v2.json');history=get('run/history.json');histories[node]=history
  assert inv['gpu'].split(',')[0]==uuid and protocol['gpu_uuid']==uuid and protocol['requested_mode']==mode
  assert protocol['formal_plan_sha256']==sha(z.read('source/formal_plan_v2.json'))
  assert protocol['epochs']==100 and protocol['seed']==91816 and not protocol['test_requested']
  for n,h in protocol['source_sha256'].items():assert sha(z.read('source/'+n))==h==plan['source_sha256'][n]
  assert sha(z.read('run/orders.npy'))==protocol['orders_sha256']==plan['orders_sha256']
  assert sha(z.read('source/teacher_cache_v1/collection.json'))==protocol['teacher_collection_sha256']
  assert [x['epoch'] for x in history]==list(range(1,len(history)+1)) and 1<=len(history)<=100
  assert all(x['effective_mode']==('fixed' if x['epoch']<=10 else mode) for x in history)
  for key in ['initial_tensor_sha256','orders_sha256','teacher_collection_sha256','jacobian_receipt_sha256']:common.setdefault(key,[]).append(protocol[key])
  if 'run/shared_phase_addon.pt' in manifest:common.setdefault('shared_phase_sha256',[]).append(manifest['run/shared_phase_addon.pt']['sha256'])
  assert protocol['jacobian_receipt_sha256']==sha(z.read('source/label_free_jacobian_v1/jacobian_receipt.json'))
  large=get('large_file_manifest.json');old=json.loads((Path(__file__).resolve().parents[1]/'outputs/连续向量三完整权重本地核验.json').read_text())
  for name,meta in large.items():
   if name.startswith(('previous_full_checkpoints/','previous_preservation/')):
    n=name.split('/')[1];assert meta['sha256']==old['rows'][n]['full_checkpoint_receipt']['full_checkpoint_sha256'] and meta['bytes']==746206408
  new_full=get_local_full=json.loads((Path(__file__).resolve().parents[1]/'outputs/Jacobian三完整权重本地核验.json').read_text(encoding='utf-8'))
  for name,meta in large.items():
   if name.startswith(('full_checkpoints/','current_preservation/')):
    training=name.split('/')[1];assert meta['sha256']==new_full['rows'][training]['full_checkpoint_receipt']['full_checkpoint_sha256'] and meta['bytes']==746206408
  cpu_sources={'a':[],'b':['a','c'],'c':['b']}[node]
  for training in cpu_sources:
   prefix='current_preservation/'+training+'/'
   cpu=get(prefix+'cpu_preservation_receipt.json');pm=get(prefix+'preservation_manifest.json')
   assert cpu['training_node']==training and cpu['target_cpu_node']==node and cpu['assembly_node']=='newA'
   assert not cpu['cuda_initialized'] and cpu['target_gpu_uuid']==uuid and len(cpu['required_seven_files'])==7
   assert cpu['source_sha256']==sha(z.read('source/verify_soft_preservation_cpu_v1.py'))
   for group in ['required_seven_files','extra_files']:
    assert cpu[group]==pm[group]
    for filename,expected_meta in pm[group].items():
     captured=(large if prefix+filename in large else manifest)[prefix+filename]
     assert captured['sha256']==expected_meta['sha256'] and captured['bytes']==expected_meta['bytes']
  state='LIVE_TRAINING'
  if 'run/selection.json' in manifest:
   sel=get('run/selection.json');assert len(history)==100 and sel['epochs']==100
   chosen=int(np.argmin([x['dev_author_batch_mean_mse'] for x in history]))+1;assert chosen==sel['best_epoch']
   assert sel['effective_mode']==('fixed' if chosen<=10 else mode)
   assert sel['addon_sha256']==manifest['run/best_addon.pt']['sha256'] and sel['prediction_sha256']==manifest['run/predictions.npz']['sha256']
   state='100_EPOCH_COMPLETE_FULL_LOCAL_VERIFICATION_SEPARATE'
   with z.open('run/predictions.npz') as f:
    arrays=np.load(__import__('io').BytesIO(f.read()));pred=arrays['valid_pred'];y=arrays['valid_y'];assert len(pred)==len(y)==229 and np.isfinite(pred).all()
    mse=float(np.mean((pred-y)**2));mae=float(np.mean(np.abs(pred-y)))
   assert abs(mse-history[chosen-1]['dev_mse_229'])<1e-6 and abs(mae-history[chosen-1]['dev_mae_229'])<1e-6
  else:
   assert inv['actual_process_argv']==inv['launch']['argv'] and inv['actual_process_argv'][1:]==protocol['argv']
   assert 'State:\tZ' not in inv['actual_process_status'] and uuid in inv['compute']
  reports[node]={'state':state,'epochs':len(history),'last_utc':history[-1]['utc'],'capture_utc':inv['utc'],'actual_argv':inv['actual_process_argv'],'gpu':inv['gpu'],'compute':inv['compute'],'disk_free':inv['data_disk_free'],'current_cpu_preservation_training_nodes':cpu_sources,'best_epoch':None if state=='LIVE_TRAINING' else chosen,'mae':None if state=='LIVE_TRAINING' else mae,'mse':None if state=='LIVE_TRAINING' else mse,'snapshot_sha256':receipt['sha256']}
for k,v in common.items():assert len(set(v))==1,k
assert histories['a'][:min(10,len(histories['a']))][0]['train_task_mse']==histories['b'][0]['train_task_mse']==histories['c'][0]['train_task_mse']
for e in range(min(10,*[len(h) for h in histories.values()])):
 for k in ['train_task_mse','train_gradient_loss','dev_author_batch_mean_mse','dev_mse_229','dev_mae_229']:assert len({h[e][k] for h in histories.values()})==1,(e,k)
out={'status':'THREE_FRESH_SNAPSHOTS_AND_NEW_CPU_PRESERVATION_INDEPENDENTLY_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reports':reports,'common':{k:v[0] for k,v in common.items()},'scope':'Fresh current seven-plus-five CPU copies crosschecked with manifests and original receipts; A-to-B/B-to-C/C-to-B targets are outside training and newA assembly. Full tensor CPU checks are proven by original verifier receipts, full original-input replay by separate prior audit.'}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(reports))

from pathlib import Path
r = Path(__file__).resolve().parent
source = (r / 'capture_soft_vector_v3.py').read_text(encoding='utf-8')
source = source.replace('capture_soft_vector_v3.py', 'capture_soft_vector_v4.py')
source = source.replace("'assets_verified.json'", "'verify_soft_preservation_cpu_v1.py','assets_verified.json'")
needle = " tree(Path('/data/coding/soft_vector_preservation_20261005T1310Z'),'previous_preservation')"
assert source.count(needle) == 1
source = source.replace(needle, needle + "\n tree(Path('/data/coding/jacobian_preservation_20261005T1406Z'),'current_preservation')")
(r / 'capture_soft_vector_v4.py').write_text(source, encoding='utf-8')
source = (r / 'audit_soft_snapshot_v3.py').read_text(encoding='utf-8')
source = source.replace('capture_soft_vector_v3.py', 'capture_soft_vector_v4.py')
source = source.replace("  state='LIVE_TRAINING'", """  new_full=get_local_full=json.loads((Path(__file__).resolve().parents[1]/'outputs/Jacobian三完整权重本地核验.json').read_text(encoding='utf-8'))
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
  state='LIVE_TRAINING'""")
source = source.replace("'100_EPOCH_ADDON_COMPLETE_FULL_CHECKPOINT_PENDING'", "'100_EPOCH_COMPLETE_FULL_LOCAL_VERIFICATION_SEPARATE'")
source = source.replace("'best_epoch':None", "'current_cpu_preservation_training_nodes':cpu_sources,'best_epoch':None")
source = source.replace("'THREE_FRESH_SNAPSHOTS_INDEPENDENTLY_AUDITED'", "'THREE_FRESH_SNAPSHOTS_AND_NEW_CPU_PRESERVATION_INDEPENDENTLY_AUDITED'")
source = source.replace('Full.pt not inferred from addon completion; final fullclassifier replay and rotated CPU preservation separately required.', 'Fresh current seven-plus-five CPU copies crosschecked with manifests and original receipts; A-to-B/B-to-C/C-to-B targets are outside training and newA assembly. Full tensor CPU checks are proven by original verifier receipts, full original-input replay by separate prior audit.')
(r / 'audit_soft_snapshot_v4.py').write_text(source, encoding='utf-8')
print('NEW_CAPTURE_V4_CREATED_WITH_NEW_PRESERVATION_ROOT')

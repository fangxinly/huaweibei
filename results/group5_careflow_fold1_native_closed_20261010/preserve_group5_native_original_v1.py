"""Serial preservation of a separately completed fresh native precheck."""
import argparse,datetime as dt,json,pathlib,shutil,sys,traceback,zipfile
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,verify_zip,seal,plan,upload,restore
from group5_publish_exact_local_v1 import publish_tree
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['publish','restore','close']);a.add_argument('--pointer',type=P,required=True);args=a.parse_args()
 p=json.loads(args.pointer.read_bytes());capture=P(p['root']);original=capture/p['archive_name']
 method=p['method'];fold=p['fold'];label=method+'-fold'+str(fold)
 if args.stage=='publish':
  remote=json.loads((capture/'remote_original_archive_receipt.json').read_bytes())
  assert digest(original)==remote['archive_SHA']==p['expected_SHA'] and original.stat().st_size==remote['bytes']==p['bytes']
  proof=verify_zip(original);root=DC.parent/('g5_native_'+label+'_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir();meta=root/'metadata';meta.mkdir()
  with zipfile.ZipFile(original) as z:
   for n in z.namelist():
    if n.endswith(('.pt','.zip','.pyc')) or '/__pycache__/' in n:continue
    t=meta/'original'/n;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(z.read(n))
  receipt=json.loads((meta/'original/out/actual_stage_receipt.json').read_bytes());cpu=json.loads((meta/'original/independent_CPU_audit.json').read_bytes())
  assert receipt['status']=='NATIVE_PRECHECK3_EXIT_PENDING_CPU_TRANSPORT' and cpu['status']=='INDEPENDENT_CPU_ORIGINAL_DIRECT_STATE_AUDIT_PASS'
  assert receipt['method']==cpu['method']==method and receipt['fold']==cpu['fold']==fold and receipt['updates']==cpu['updates']==3
  assert receipt['checkpoint']['SHA']==cpu['checkpoint_SHA']
  members=json.loads((meta/'original/member_manifest.json').read_bytes());assert next(x['sha256'] for x in members if x['name']=='out/precheck_full.pt')==cpu['checkpoint_SHA']
  assert not receipt['outer_labels_decoded'] and not receipt['inner_labels_decoded'] and not cpu['new_solve_fit_score_or_outer_label_decode']
  assert json.loads((meta/'original/wrapper_exit.json').read_bytes())['natural_exit']==json.loads((meta/'original/CPU_exit_observed.json').read_bytes())['natural_exit']==0
  shutil.copyfile(capture/'remote_original_archive_receipt.json',meta/'remote_original_archive_receipt.json');shutil.copyfile(P(__file__),meta/P(__file__).name)
  write(meta/'local_original_verification.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),archive_SHA=digest(original),bytes=original.stat().st_size,**proof))
  ranges=plan(original,'group5-'+label+'-native-'+p['expected_SHA'][:12]+'.zip');write(root/'range_manifest.json',ranges)
  p.update(preservation_root=str(root),local_verification=proof);write(args.pointer,p)
  upload(ranges,root/'Release_ranges_receipt.json');print('NATIVE_ORIGINAL_PARTS_VERIFIED_RESTORE_PENDING',flush=True)
 elif args.stage=='restore':
  root=P(p['preservation_root']);ranges=json.loads((root/'range_manifest.json').read_bytes());assert shutil.disk_usage('C:/').free>ranges['bytes']+200*1024**2
  result=restore(ranges,json.loads((root/'Release_ranges_receipt.json').read_bytes()),capture/'GitHub_restored_complete_original.zip');write(root/'actual_GitHub_restoration.json',result);print(json.dumps(result),flush=True)
 else:
  root=P(p['preservation_root']);meta=root/'metadata';closed=json.loads((root/'actual_GitHub_restoration.json').read_bytes());assert closed['whole_SHA']==p['expected_SHA'] and closed['all_member_SHA_CRC_unique_exact_set_passed']
  native=json.loads((meta/'original/out/actual_stage_receipt.json').read_bytes());native_plan=json.loads((meta/'original/plan.json').read_bytes())
  for n in ['range_manifest.json','Release_ranges_receipt.json','actual_GitHub_restoration.json']:shutil.copyfile(root/n,meta/n)
  qualification=dict(status='SAME_METHOD_FOLD_NATIVE3_CPU_FULL_RELEASE_RESTORE_CLOSED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),method=method,fold=fold,
   original_archive_SHA=p['expected_SHA'],bytes=p['bytes'],CPU_original_exit=0,native_original_exit=0,updates=3,
   checkpoint_SHA=native['checkpoint']['SHA'],original_native_receipt_SHA=digest(meta/'original/out/actual_stage_receipt.json'),
   clean_initial_state_SHA=native['clean_initial_state_SHA'],initial_rng_SHA=native['initial_rng_SHA'],projected_fit_seconds=native['projected_fit_seconds'],
   source_SHA=native_plan['source_SHA'],original_plan_SHA=digest(meta/'original/plan.json'),split_SHA=native_plan['split_SHA'],
   same_method_fold_native_CPU_qualified=True,Release_range_restore_qualified=True,fresh_initial_state_qualified=True,
   all_member_SHA_CRC_unique_exact_set_passed=True,actual_GitHub_restoration=closed,original_once_not_repeated=True,
   formal_training_dispatched=False,outer_scores_computed=False,remote_root=p['remote'],permanent_remote_original=p['remote']+'/'+p['archive_name'])
  write(meta/'qualification_receipt.json',qualification);proof=seal(meta,meta/'complete_native_preservation_metadata.zip');write(root/'D_receipt.json',proof)
  metadata_ranges=plan(P(proof['archive']),'group5-'+label+'-native-closed-'+proof['archive_SHA'][:12]+'.zip')
  write(root/'metadata_range_manifest.json',metadata_ranges)
  upload(metadata_ranges,root/'metadata_Release_receipt.json')
  assert shutil.disk_usage('D:/').free>proof['bytes']+40*1024**2
  metadata_restored=restore(metadata_ranges,json.loads((root/'metadata_Release_receipt.json').read_bytes()),root/'GitHub_restored_native_metadata.zip')
  write(root/'metadata_actual_GitHub_restoration.json',metadata_restored)
  folder=root/'text_publication';folder.mkdir();binary=[]
  for f in sorted(meta.rglob('*')):
   if not f.is_file():continue
   rel=f.relative_to(meta)
   if f.suffix in ('.zip','.npy','.npz'):
    binary.append(dict(path=rel.as_posix(),bytes=f.stat().st_size,SHA=digest(f)));continue
   raw=f.read_bytes();raw.decode('utf-8-sig');target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  for n in ['metadata_range_manifest.json','metadata_Release_receipt.json','metadata_actual_GitHub_restoration.json']:
   shutil.copyfile(root/n,folder/n)
  write(folder/'binary_original_Release_scope.json',dict(original_numeric_evidence_in_unchanged_Release=binary,original_archive_SHA=p['expected_SHA'],complete_original_GitHub_restoration=closed))
  github=publish_tree(folder,'results/group5_'+method+'_fold'+str(fold)+'_native_closed_20261010','Preserve fresh '+label+' native3 original CPU audit and complete Release restoration; source reports indices in Git')
  write(root/'GitHub_receipt.json',github)
  state=json.loads((DC/'D_current_research_state.json').read_bytes());g=state['latest_human_TEST_selected_group5'];g['latest_'+method+'_fold'+str(fold)+'_native_closed']=dict(root=str(root),qualification=qualification,preservation=proof,github=github)
  state.update(updated_at_utc=qualification['actual_UTC'],github_source=github,next_gate='Fresh '+label+' native3 CPU/full original transport closed. Freeze a distinct formal stage only if measured total training budget plus at least2h saving fits fresh lease/resources. No repeated native/old once or OUTER scoring.');sync(state)
  for name in ['GitHub_restored_complete_original.zip',p['archive_name']]:
   f=capture/name;assert f.resolve().parent==capture.resolve() and digest(f)==p['expected_SHA'];f.unlink()
  p.update(qualification=qualification,github=github,new_temporary_transit_copies_removed=True,remote_and_GitHub_originals_preserved=True);write(args.pointer,p)
  print(json.dumps(dict(root=str(root),qualification=qualification,github=github)),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  q=None
  if '--pointer' in sys.argv:q=P(sys.argv[sys.argv.index('--pointer')+1])
  if q and q.exists():
   p=json.loads(q.read_bytes());root=P(p.get('preservation_root',p['root']))
   write(root/('publication_failure_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),dict(observed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),original_preserved=True,new_scientific_execution=False))
  print(type(exc).__name__,str(exc),file=sys.stderr);raise

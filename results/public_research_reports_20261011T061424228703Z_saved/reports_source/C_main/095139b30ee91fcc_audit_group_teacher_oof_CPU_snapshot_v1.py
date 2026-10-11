from pathlib import Path
import argparse,json,hashlib,zipfile,datetime
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--oof',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda raw:hashlib.sha256(raw).hexdigest();r=json.loads((a.directory/'receipt.json').read_text());raw=(a.directory/'snapshot.zip').read_bytes();assert len(raw)==r['bytes'] and sha(raw)==r['sha256'] and (a.directory/'exit_code.txt').read_text().strip()=='0'
cpu=json.loads((a.oof/'original_B_CPU_preservation_receipt.json').read_text());manifest=json.loads((a.payload/'manifest.json').read_text());assert cpu['target_node']=='b' and cpu['target_gpu_uuid']=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067' and cpu['files']==manifest['files']
assert cpu['status']=='INDEPENDENT_B_CPU_COMBINED1281_OOF_PAYLOAD_SHA_CRC_ORIGINAL_OUTER_ROWS_LABELS_BASELINE_METRICS_VERIFIED'
assert cpu['transport_manifest_sha256']==sha((a.payload/'manifest.json').read_bytes()) and cpu['transport_sha256']==manifest['sha256']==sha((a.payload/'payload.zip').read_bytes())
assert json.loads((a.oof/'original_B_CPU_audit_exit.json').read_text())['exit_code']==0
assert cpu['array_audit_sha256']==sha((a.oof/'original_B_CPU_oof_array_audit.json').read_bytes())
audit=json.loads((a.oof/'original_B_CPU_oof_array_audit.json').read_text());local=json.loads((a.oof/'analysis.json').read_text());assert audit['OOF_npz_sha256']==sha((a.oof/'train_scalar_oof.npz').read_bytes()) and audit['original_analysis_sha256']==sha((a.oof/'analysis.json').read_bytes())
assert audit['metrics']['teacher_mse']==local['teacher_mse'] and audit['teacher_better_videos']==52
prefix='group_teacher/cpu_preservation/oof_20261006T031527Z/'
with zipfile.ZipFile(a.directory/'snapshot.zip') as z,zipfile.ZipFile(a.previous/'snapshot.zip') as old:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));members=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(members)|{'member_manifest.json'}
 for n,meta in members.items():assert len(z.read(n))==meta['bytes'] and sha(z.read(n))==meta['sha256']
 for name,meta in json.loads(old.read('member_manifest.json')).items():
  if name.endswith('inventory.json') or name=='large_file_manifest.json':continue
  assert members[name]==meta,('Previously completed evidence changed',name)
 large=json.loads(z.read('large_file_manifest.json'))
 for name,meta in json.loads(old.read('large_file_manifest.json')).items():assert large[name]['bytes']==meta['bytes'] and large[name]['sha256']==meta['sha256']
 for n,meta in cpu['files'].items():assert members[prefix+n]==meta
 for remote,local_name in [('oof_CPU_preservation_receipt.json','original_B_CPU_preservation_receipt.json'),('original_CPU_oof_array_audit.json','original_B_CPU_oof_array_audit.json'),('CPU_oof_audit_exit.json','original_B_CPU_audit_exit.json'),('CPU_oof_audit.log','original_B_CPU_audit.log')]:assert z.read(prefix+remote)==(a.oof/local_name).read_bytes()
 assert sha(z.read(prefix+'audit_group_teacher_oof_arrays_v1.py'))==cpu['OOF_auditor_sha256']
 assert sha(z.read('source/capture_soft_vector_v19.py'))=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
out={'status':'COMBINED1281_OOF_INDEPENDENT_CPU_ORIGINAL_RECEIPTS_AND_NEW_REAL_B_ATOMIC_SNAPSHOT_JOINED_VERIFIED_OLD_COMPLETED_EVIDENCE_UNCHANGED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_B_capture_utc':r['utc'],'snapshot_sha256':r['sha256'],'CPU_original_receipt_sha256':sha((a.oof/'original_B_CPU_preservation_receipt.json').read_bytes()),'OOF_npz_sha256':audit['OOF_npz_sha256'],'payload_members_verified':len(cpu['files']),'teacher_mse':local['teacher_mse'],'teacher_better_videos':52,'fullweights_retransmitted':False,'whole_pipeline_crossfit':False}
assert not a.out.exists();a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out['status'])

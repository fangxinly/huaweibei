from pathlib import Path
import json,zipfile,hashlib,datetime
base=Path('D:/CodexBackups/selective_flow_20261003_1105');shots=base/'finite_c2_completed_snapshots_20261005T1627Z';done=base/'finite_risk_completed_20261005';rows={};sha=lambda b:hashlib.sha256(b).hexdigest()
for node in ['a','b','c']:
 with zipfile.ZipFile(shots/node/'snapshot.zip') as z:
  get=lambda n:json.loads(z.read(n));m=get('member_manifest.json');large=get('large_file_manifest.json');inv=get('inventory.json');assert inv['capture_source_sha256']==sha(z.read('source/capture_soft_vector_v14.py'))==sha(Path('work/capture_soft_vector_v14.py').read_bytes())
  trainings=['a','b'] if node=='a' else []
  for train in trainings:
   r=get('finite_full_checkpoints/'+train+'/full_checkpoint_receipt.json');local=json.loads((done/train/'full_checkpoint_receipt.json').read_text());assert r==local
   x=large['finite_full_checkpoints/'+train+'/full_checkpoint.pt'];assert x['sha256']==r['full_checkpoint_sha256'] and x['bytes']==r['full_checkpoint_bytes']
  cpus={'a':[],'b':['a'],'c':['b']}[node]
  for train in cpus:
   pre='finite_preservation/'+train+'/';cpu=get(pre+'cpu_preservation_receipt.json');local=json.loads((done/train/'cpu_preservation_receipt.json').read_text());pm=get(pre+'preservation_manifest.json');assert cpu==local and cpu['target_cpu_node']==node
   for group in ['required_seven_files','extra_files']:
    assert pm[group]==cpu[group]
    for n,x in pm[group].items():
     captured=(large if pre+n in large else m)[pre+n];assert captured['sha256']==x['sha256'] and captured['bytes']==x['bytes']
  if node=='a':
   original=json.loads((base/'finite_risk_preflight_20261005T1434Z/a/single_token_full_raw_receipt.json').read_text());assert get('finite_single_token_precheck/full_raw_receipt.json')==original
  if node=='c':
   for filename,local in [('mechanism_receipt.json','single_token_mechanism_receipt.json'),('warmup_receipt.json','single_token_warmup_receipt.json')]:assert get('finite_single_token_precheck/'+filename)==json.loads((base/'finite_risk_preflight_20261005T1434Z/c'/local).read_text())
   assert get('finite_failed_diagnostics/reproduction.json')['first_nonfinite_batch']==13 and get('finite_failed_diagnostics/zero_norm_probe_v2.json')['records'][0]['row_ids']==[620]
   assert 'DivBackward0' in z.read('finite_failed_diagnostics/anomaly_trace.txt').decode()
  rows[node]=dict(capture_utc=inv['utc'],new_full_training_nodes=trainings,new_cpu_training_nodes=cpus,large_only_fresh_hash_references=True)
out=dict(status='FINITE_AB_FULL_CPU_AND_CFAILURE_REPAIR_SNAPSHOTS_REFERENCES_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope='Large fullweights only fresh SHA references; originals already permanentD/independentCPU. Only new A/B complete; Cfailed retained, singleton prechecks not100formal.')
Path('outputs/有限任务风险C2完成后旧保存联合快照核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rows))

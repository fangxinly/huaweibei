from pathlib import Path
import json,hashlib,zipfile,datetime
w=Path(__file__).parent;root=w.parent;base=Path('D:/CodexBackups/selective_flow_20261003_1105');shots=base/'finite_c2_completed_snapshots_20261005T1627Z';done=base/'finite_c2_completed_20261005/c';sha=lambda b:hashlib.sha256(b).hexdigest()
local=json.loads((root/'outputs/有限任务风险C2完整权重本地独立核验.json').read_text(encoding='utf-8'));cpu=json.loads((done/'cpu_preservation_receipt.json').read_text(encoding='utf-8'));rows={}
for node in ['a','b','c']:
 with zipfile.ZipFile(shots/node/'snapshot.zip') as z:
  get=lambda n:json.loads(z.read(n));m=get('member_manifest.json');large=get('large_file_manifest.json');inv=get('inventory.json');assert inv['capture_source_sha256']==sha(z.read('source/capture_soft_vector_v14.py'))==sha((w/'capture_soft_vector_v14.py').read_bytes())
  if node=='a':
   pre='finite_c2_assembly/full_checkpoints/c/';r=get(pre+'full_checkpoint_receipt.json');assert r==local['receipt']
   manifest=get(pre+'preservation_manifest.json');assert manifest==local['manifest']
  elif node=='b':
   pre='finite_c2_preservation/c/';assert get(pre+'cpu_preservation_receipt.json')==cpu
   manifest=get(pre+'preservation_manifest.json');assert manifest==local['manifest'] and cpu['target_cpu_node']=='b' and cpu['training_node']=='c' and cpu['assembly_node']=='a'
  else:
   second=get('finite_single_token_secondorder/receipt.json');original=json.loads((base/'finite_followup_20261005T1556Z/c/secondorder_receipt.json').read_text(encoding='utf-8'));assert second==original and second['double_unrolled_finite_difference']['error']<1e-8
   manifest=None
  if manifest:
   for group in ['required_seven_files','extra_files']:
    for n,x in manifest[group].items():
     v=(large if pre+n in large else m)[pre+n];assert v['bytes']==x['bytes'] and v['sha256']==x['sha256']
  rows[node]=dict(capture_utc=inv['utc'],full_checkpoint_assembly_reference=node=='a',original_cpu_receipt_verified=node=='b',c2_formal_and_secondorder_reference=node=='c',large_fullweights_fresh_sha_only=True)
out=dict(status='C2_FULL_STRICTREPLAY_PERMANENTD_AND_BCPU_ORIGINAL_RECEIPTS_COMBINED_SNAPSHOTS_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope='Actual source/manifest/receipt proofs. Largefullweights freshSHA only, already downloaded separately. No oldweight retransfer or deletion.')
(root/'outputs/有限任务风险C2完整模型独立CPU及最终联合快照核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rows))

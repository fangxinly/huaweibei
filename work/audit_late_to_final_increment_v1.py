from pathlib import Path
import argparse,json,zipfile,hashlib,datetime
p=argparse.ArgumentParser();p.add_argument('--late',type=Path,required=True);p.add_argument('--final',type=Path,required=True);a=p.parse_args()
dynamic={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'}
rows={}
for n in 'abc':
 with zipfile.ZipFile(a.late/n/'snapshot.zip') as old,zipfile.ZipFile(a.final/n/'snapshot.zip') as new:
  om=json.loads(old.read('member_manifest.json'));m=json.loads(new.read('member_manifest.json'))
  ol=json.loads(old.read('large_file_manifest.json'));l=json.loads(new.read('large_file_manifest.json'))
  changed=[];same=0
  for k,it in om.items():
   if k in dynamic:continue
   if m[k]==it:same+=1;continue
   assert k.endswith('.log') and it['bytes']==0 and k.split('/')[-1].startswith('capture_')
   b=new.read(k).decode();assert b.startswith('NEW_RESEARCH_CAPTURE_COMPLETE ') and len(b.splitlines())==1
   changed.append(k)
  assert set(l)==set(ol)
  for k,it in ol.items():assert l[k]['sha256']==it['sha256'] and l[k]['bytes']==it['bytes']
  rows[n]={'unchanged_scientific_and_preservation_members':same,'natural_stdout_closure_only':changed,'all_large_references_unchanged':len(l),'new_members':sorted(set(m)-set(om))}
proof={'status':'ACTUAL_LATE_TO_FINAL_WINDOW_ONLY_NEW_PRESERVATION_ROOTS_AND_NATURAL_STDOUT_CLOSURE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'fresh_D_teacher_full_CPU_joint_audit':str(a.late/'fresh_D_teacher_full_CPU_snapshot_joint_audit.json'),'fresh_D_teacher_audit_sha256':hashlib.sha256((a.late/'fresh_D_teacher_full_CPU_snapshot_joint_audit.json').read_bytes()).hexdigest(),'large_weight_references_not_new_downloads':True}
assert not (a.final/'late_to_final_increment_audit.json').exists()
(a.final/'late_to_final_increment_audit.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(proof['status'])

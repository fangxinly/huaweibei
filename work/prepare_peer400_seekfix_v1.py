"""Reuse exact already publicly downloaded archive; temporary PT avoids outer-ZIP seek amplification."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
for node in ('A','B'):
 old=base/'work'/(node+'_interrupted_peer400_qualified_20261009T001701Z');root=base/'work'/(node+'_interrupted_peer400_seekfix_qualified_'+a.tag);root.mkdir()
 for f in old.iterdir():
  if f.suffix!='.zip':shutil.copy2(f,root/f.name)
 f=root/'audit_interrupted_candidate400_v1.py';s=f.read_text(encoding='utf8')
 s=s.replace("p.add_argument('--uuid',required=True);", "p.add_argument('--retained-archive',type=Path,required=True);p.add_argument('--uuid',required=True);")
 start=s.index("a.out.mkdir();archive=");end=s.index("assert digest(archive)==",start)
 s=s[:start]+"a.out.mkdir();archive=a.retained_archive\nassert archive.resolve().is_relative_to(Path('/data/coding'))\n"+s[end:]
 s=s.replace("with z.open('prior/out/complete_resume_step400.pt') as f:state=torch.load(f,map_location='cpu')", "temporary=a.out/'_temporary_saved400.pt'\n with z.open('prior/out/complete_resume_step400.pt') as inp,temporary.open('xb') as out:\n  for block in iter(lambda:inp.read(8*1024**2),b''):out.write(block)\n assert temporary.stat().st_size==cap['warm400_checkpoint_bytes'] and digest(temporary)==cap['warm400_checkpoint_SHA']\n state=torch.load(temporary,map_location='cpu')")
 s=s.replace('public_Release_download=True,all_member_SHA_CRC_unique=True','public_Release_download=False,retained_public_Release_origin_archive_reuse=True,temporary_PT_bytes_SHA_verified=True,all_member_SHA_CRC_unique=True')
 marker="(a.out/'audit_result.json').write_text"
 s=s.replace(marker,"# This task-owned temporary extraction is explicitly disposable; immutable original ZIP retained.\nassert temporary.resolve().parent==a.out.resolve() and temporary.name=='_temporary_saved400.pt' and digest(archive)==cap['archive_SHA']\ntemporary.unlink();r['temporary_PT_removed_after_success']=True\n"+marker)
 f.write_text(s,encoding='utf8');ast.parse(s)
 wrapper=root/'run_interrupted_peer400_capture_v1.py';s=wrapper.read_text(encoding='utf8');s=s.replace("'--peer',q['source_peer']]", "'--peer',q['source_peer'],'--retained-archive',q['retained_archive']]")
 s=s.replace("f.name!='peer_interrupted_original.zip'", "f.name not in ('peer_interrupted_original.zip','_temporary_saved400.pt')")
 s=s.replace("files={f.relative_to(a.root)", "write(a.root/'large_original_references.json',dict(retained_public_original=q['retained_archive'],temporary_PT_disposable_after_success=q['temporary_checkpoint_extraction_is_disposable_after_success'],temporary_PT_retained_on_failure=code!=0))\nfiles={f.relative_to(a.root)")
 wrapper.write_text(s,encoding='utf8');ast.parse(s)
 q=json.loads((root/'plan.json').read_text());q.update(status='INTERRUPTED_PEER400_RETAINED_PUBLIC_ORIGINAL_SEEKFIX_QUALIFIED',actualclock_before_qualification_UTC=a.stamp,retained_archive='/data/coding/'+node+'_interrupted_peer400_actual_20261009T001701Z/out/peer_interrupted_original.zip',temporary_checkpoint_extraction_is_disposable_after_success=True,no_old_frozen_original_delete=True,prior_stream_audit_not_claimed_complete=True,source_sha256={f.name:sha(f) for f in root.glob('*.py')})
 (root/'plan.json').write_text(json.dumps(q,indent=2),encoding='utf8')
 with zipfile.ZipFile(root/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in root.iterdir():
   if f.suffix!='.zip':z.write(f,f.name)
 print(json.dumps(dict(node=node,root=str(root),plan_SHA=sha(root/'plan.json'),source_SHA=sha(root/'source.zip'))))

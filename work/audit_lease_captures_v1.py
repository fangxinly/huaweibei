from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--deadline',required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--new-source',type=Path);c=a.parse_args()
deadline=datetime.datetime.fromisoformat(c.deadline);assert deadline.tzinfo is not None
w=Path(__file__).resolve().parent;original=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/outputs/parallel_gpu_20261003_1800/capture_repeat_live_20261004T0658Z.py')
sources={'old':hashlib.sha256(original.read_bytes()).hexdigest(),'new':hashlib.sha256((c.new_source or w/'capture_counterfactual_followup_v3.py').read_bytes()).hexdigest()}
uuids={'a':'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3','b':'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
rows=[]
for node,uuid in uuids.items():
 for kind in ('old','new'):
  d=c.directory/node/kind;proof=json.loads((d/'proof.json').read_text());raw=(d/'snapshot.zip').read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==proof['archive_sha256']
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   manifestname='manifest.json' if kind=='old' else 'member_manifest.json';manifest=json.loads(z.read(manifestname));assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(manifest)|{manifestname}
   for name,e in manifest.items():
    b=z.read(name);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
   if kind=='old':
    actual=json.loads(z.read('actual.json'));assert proof['actual']==actual and proof['members']==len(manifest) and actual['node']=='p4_'+node and actual['capture_source_sha256']==sources[kind]
    gpu=actual['gpu'];compute=actual['compute'];ps=actual['ps'];free=actual['free_bytes']
   else:
    actual=json.loads(z.read('actual_resources.json'));assert proof['members']==manifest and proof['archive_bytes']==len(raw) and proof['capture_source_sha256']==sources[kind]
    assert all(actual[k]['returncode']==0 for k in ['gpu','compute','processes']);gpu=actual['gpu']['output'];compute=actual['compute']['output'];ps=actual['processes']['output'];free=actual['disk_free_bytes']
    assert actual['captured_at']==proof['captured_at']
   when=datetime.datetime.fromisoformat(actual['captured_at']);lag=(when-deadline).total_seconds();assert 0<=lag<=120,(node,kind,lag)
   assert uuid in gpu and free>0 and ps.strip()
   rows.append({'node':node,'kind':kind,'captured_at':actual['captured_at'],'seconds_after_deadline':lag,'archive_sha256':digest,'archive_bytes':len(raw),'members':len(manifest),'capture_source_sha256':sources[kind],'gpu':gpu,'compute':compute,'free_bytes':free,'ps_sha256':hashlib.sha256(ps.encode()).hexdigest(),'top_level_roots':sorted({n.split('/')[0] for n in manifest if '/' in n})})
r={'audited_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline':c.deadline,'status':'SIX_REAL_DEADLINE_OLD_AND_NEW_CAPTURES_ALL_SHA_VERIFIED','rows':rows,'limits':'Old named capture has three named roots only; new13 roots. Full weights separately preserved, not in these archives. Metadata individual reads, not atomic.'}
assert not c.out.exists();c.out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':r['status'],'rows':len(rows)}))

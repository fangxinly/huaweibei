import hashlib,json,tarfile,datetime,subprocess,shutil
from pathlib import Path,PurePosixPath
base=Path('/data/coding/selective_flow');p=base/'provision_20261004T0647Z';m=json.loads((p/'manifest.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert (p/'inputs.tar').stat().st_size==m['archive_bytes'] and sha(p/'inputs.tar')==m['archive_sha256']
assert not (base/'strong_baselines').exists() and shutil.disk_usage(base).free>3_000_000_000
with tarfile.open(p/'inputs.tar') as z:
 for t in z:
  parts=PurePosixPath(t.name).parts;assert parts[0]=='strong_baselines' and '..' not in parts
  if t.issym() or t.islnk():assert t.name in m['links'] and t.linkname==m['links'][t.name]['target']
 z.extractall(base)
for n,v in m['files'].items():assert sha(base/n)==v['sha256'] and (base/n).stat().st_size==v['bytes'],n
for n,v in m['links'].items():assert (base/n).is_symlink() and str((base/n).readlink())==v['target'],n
B=base/'strong_baselines';r=subprocess.run([str(B/'.venv/bin/python'),'-c',"import torch,transformers,numpy,sentencepiece; assert torch.__version__=='2.1.0+cu121' and transformers.__version__=='4.37.2' and numpy.__version__=='1.26.4'; print(torch.__version__,transformers.__version__,numpy.__version__)"],capture_output=True,text=True)
assert r.returncode==0,r.stderr
assert subprocess.check_output(['git','-C',str(B/'CaReFlow'),'rev-parse','HEAD'],text=True).strip()==m['reference_protocol']['author_commit']
proof={'status':'PORTABLE_INPUTS_ALL_FILES_SHA_AND_RUNTIME_VERIFIED','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':m['archive_sha256'],'files_verified':len(m['files']),'runtime':r.stdout,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'free_bytes':shutil.disk_usage(B).free}
with (p/'destination_verification.json').open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof),flush=True)

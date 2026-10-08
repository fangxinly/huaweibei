"""Fresh capture of new resources only; completion states remain distinct."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,zipfile,shutil
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
out=a.deployment/('capture_'+a.stamp);out.mkdir(exist_ok=False)
q=lambda args:subprocess.check_output(args,text=True).strip()
launch=json.loads((a.deployment/'launch.json').read_text());proc=Path('/proc')/str(launch['pid']);argv=None;status=None
if (proc/'cmdline').exists():argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1];status=(proc/'status').read_text()
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'capture_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gpu':q(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),'compute':q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),'launch':launch,'actual_process_argv':argv,'actual_process_status':status,'data_disk_free':shutil.disk_usage('/data').free,'root':str(a.root),'deployment':str(a.deployment),'selection_present':(a.deployment/'run/selection.json').exists(),'full_checkpoint_present':(a.deployment/'run/full_checkpoint.pt').exists()}
(out/'inventory.json').write_text(json.dumps(r,indent=2))
(out/'pip_freeze.txt').write_text(q(['python','-m','pip','freeze']))
members={};zpath=out/'snapshot.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 def add(path,name):
  b=path.read_bytes();z.writestr(name,b);members[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 for path in out.iterdir():
  if path.suffix!='.zip':add(path,path.name)
 for n in ['formal_plan_v1.json','train_soft_vector_v1.py','soft_vector_runtime_v1.py','task_gradient_vector_candidate_v3.py','check_soft_full_inference_v1.py','assets_verified.json','teacher_cache_v1/collection.json','teacher_cache_v1/train_gradient_rms.npy']:
  add(a.root/n,'source/'+n)
 for path in sorted(a.deployment.glob('*.json')):add(path,path.name)
 for path in sorted((a.deployment/'run').iterdir()):
  if path.name!='full_checkpoint.pt' and path.suffix in ['.json','.npz','.npy','.pt']:add(path,'run/'+path.name)
  elif path.name=='full_checkpoint.pt':r['full_checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 z.writestr('member_manifest.json',json.dumps(members,indent=2))
receipt={'utc':r['utc'],'sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),'bytes':zpath.stat().st_size,'members':len(members),'full_checkpoint_sha256':r.get('full_checkpoint_sha256'),'snapshot':str(zpath)}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('CAPTURE_COMPLETE',a.stamp,len(members),flush=True)

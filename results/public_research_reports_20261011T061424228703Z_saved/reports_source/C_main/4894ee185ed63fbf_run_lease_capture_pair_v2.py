from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
a=argparse.ArgumentParser();a.add_argument('--node',choices=['a','b','c'],required=True);a.add_argument('--stamp',required=True);a.add_argument('--deadline',required=True);a.add_argument('--new-source-sha',required=True);a.add_argument('--uuid',required=True);c=a.parse_args()
assert c.stamp.isalnum();deadline=datetime.datetime.fromisoformat(c.deadline);now=datetime.datetime.now(datetime.timezone.utc);assert 0<=(now-deadline).total_seconds()<=90
base=Path('/data/coding/selective_flow');python=base/'strong_baselines/.venv/bin/python'
old=base/'capture_repeat_live_20261004T0658Z.py';new=base/'capture_counterfactual_followup_v4_20261005T0545Z.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='eacbc5e3b7ad2122a3d7d2017ecfa96220497ab335b5687c48836f9048743c0a'
assert hashlib.sha256(new.read_bytes()).hexdigest()==c.new_source_sha
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert uuid==c.uuid
out=base/('lease_pair_'+c.stamp+'_'+c.node);out.mkdir(exist_ok=False);records=[]
for name,args,proof in [('old',[str(python),str(old),'--node','p4_'+c.node,'--stamp',c.stamp],base/('repeat_snapshot_'+c.stamp)/('p4_'+c.node)/'proof.json'),('new',[str(python),str(new),'--stamp',c.stamp],base/('continuation_capture_'+c.stamp)/'proof.json')]:
 with (out/(name+'.log')).open('xb') as f:r=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode)
 records.append({'kind':name,'returncode':r.returncode,'args':args,'proof_path':str(proof),'proof_sha256':hashlib.sha256(proof.read_bytes()).hexdigest()})
report={'deadline':c.deadline,'started_at':now.isoformat(),'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':c.node,'gpu_uuid':uuid,'records':records,'capture_pair_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (out/'pair_receipt.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))

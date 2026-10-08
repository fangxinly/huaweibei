"""Record the actual natural exit of the existing frozen atomic capture."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
base=Path('/data/coding/soft_vector_research_20261005T1220Z')
deployment=Path('/data/coding/jacobian_aligned_v2_deployment_20261005T1327Z')
source=base/'capture_soft_vector_v19.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
command=[sys.executable,str(source),'--root',str(base),'--deployment',str(deployment),'--stamp',a.stamp]
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
start=now()
log=a.root/('capture_'+a.stamp+'.log')
with log.open('x') as handle:
    child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
    code=child.wait()
out=deployment/('capture_'+a.stamp)
proof={'started_utc':start,'finished_utc':now(),'child_pid':child.pid,'argv':command,'exit_code':code,
       'capture_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'wrapper_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if code==0:
    assert 'NEW_RESEARCH_CAPTURE_COMPLETE' in log.read_text()
    assert (out/'receipt.json').is_file() and (out/'snapshot.zip').is_file()
(out/'exit_code.txt').write_text(str(code)+'\n')
(out/'actual_capture_exit.json').write_text(json.dumps(proof,indent=2))
print('ACTUAL_CAPTURE_EXIT',code,a.stamp,flush=True)
raise SystemExit(code)

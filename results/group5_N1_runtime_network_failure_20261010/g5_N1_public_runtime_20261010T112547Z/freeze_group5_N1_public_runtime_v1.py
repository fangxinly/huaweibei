"""Freeze isolated public runtime setup only on the authorized new N1 node."""
import ast,datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC

def main():
 now=dt.datetime.now(dt.timezone.utc);root=DC.parent/('g5_N1_public_runtime_plan_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir();q=root/'payload';q.mkdir()
 t=json.loads((BASE/'work/group5_idle_public_runtime_pointer.json').read_bytes());p=json.loads((P(t['root'])/'prepared_runtime_template.json').read_bytes())
 assert digest(P(t['root'])/'prepared_runtime_template.json')==t['template_SHA']
 remote='/data/coding/g5_N1_public_runtime_'+now.strftime('%Y%m%dT%H%M%SZ')
 p.update(status='GROUP5_BATCH5_PUBLIC_RUNTIME_ONLY_FROZEN',GPU_UUID='GPU-cf38498b-a118-7e1a-45f1-0b5502b36cde',lease_end_UTC='2026-10-10T16:30:58+00:00',stage_budget_seconds=900,
  local_C_free_bytes=shutil.disk_usage('C:/').free,local_D_free_bytes=shutil.disk_usage('D:/').free,local_space_capture_UTC=now.isoformat(),
  scientific_dispatch_authorized_by_this_plan=False,no_task_training_inference_score=True)
 source=BASE/'work/group5_public_runtime_v1.py';ast.parse(source.read_text(encoding='utf8'));assert digest(source)==p['source_SHA'];shutil.copyfile(source,q/source.name)
 write(q/'plan.json',p);h=digest(q/'plan.json')
 wrapper="import subprocess,json,pathlib,datetime\nr=pathlib.Path("+repr(remote)+")\na="+repr(['/data/miniconda/envs/torch/bin/python','-B',remote+'/group5_public_runtime_v1.py','--plan',remote+'/plan.json','--plan-sha',h,'--root',remote+'/runtime'])+"\nwith (r/'stdout.log').open('xb') as f:\n c=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT)\n (r/'dispatch.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,argv=a)))\n code=c.wait()\n(r/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,natural_exit=code)))\nraise SystemExit(code)\n"
 ast.parse(wrapper);(q/'wrapper.py').write_text(wrapper,encoding='utf8');shutil.copyfile(P(__file__),q/P(__file__).name)
 proof=seal(q,q/'public_runtime_plan_original.zip');write(root/'D_plan_receipt.json',proof)
 publish(P(proof['archive']),proof['archive_SHA'],'group5-N1-runtime-plan-'+proof['archive_SHA'][:12]+'.zip',root/'Release_plan_receipt.json')
 v=dict(root=str(root),remote=remote,plan_SHA=h,Release=json.loads((root/'Release_plan_receipt.json').read_bytes()),proof=proof,SSH_session=39320,scientific_training_dispatched=False)
 write(BASE/'work/group5_N1_public_runtime_pointer.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':main()

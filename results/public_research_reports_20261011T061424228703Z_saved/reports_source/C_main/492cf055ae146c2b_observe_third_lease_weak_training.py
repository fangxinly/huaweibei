"""Read one real running prefix; never change or stop the training worker."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());assert not a.output.exists()
def load(path):return json.loads(path.read_text()) if path.exists() else None
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
root=Path(plan['root']);execution=Path(plan['execution']);bundle=Path(plan['bundle']);out=root/'out'
assert sha(bundle/'training_execution_protocol.json')==plan['protocol_SHA']
for name,h in load(bundle/'training_execution_protocol.json')['source_sha256'].items():assert sha(bundle/name)==h
child=load(execution/'actual_child.json');natural=load(execution/'natural_exit.json')
assert child['protocol_sha256']==plan['protocol_SHA']
steps=out/'actual_FIT_steps.jsonl';records=steps.read_text().splitlines() if steps.exists() else []
history=[]
for line in (execution/'stdout.log').read_text().splitlines():
    try:item=json.loads(line)
    except ValueError:continue
    if isinstance(item,dict) and 'epoch' in item:history.append(item)
err=(execution/'stderr.log').read_text()
result=dict(status='ACTUAL_SINGLE_FIXED40_RUNNING_PREFIX_OBSERVATION' if natural is None else 'ACTUAL_SINGLE_FIXED40_NATURAL_EXIT_OBSERVATION',
    actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),operator_pid=os.getpid(),operator_fullargv=[sys.executable]+sys.argv,
    child=child,natural_exit=natural,training_root=str(root),training_protocol_SHA=plan['protocol_SHA'],
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),
    full_process_table=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),
    remote_free_bytes=shutil.disk_usage(root).free,actual_completed_epoch=len(history),actual_step_prefix=len(records),
    last_completed_epoch=history[-1] if history else None,last_FIT_step=json.loads(records[-1]) if records else None,
    original_construction=load(out/'actual_construction.json'),original_physical_preflight=load(out/'actual_physical_preflight.json'),
    stderr_has_Traceback='Traceback' in err,stderr_original_tail=err[-12000:],
    actual_stage_receipt=load(out/'actual_stage_receipt.json'),observer_source_SHA=sha(Path(__file__)),
    prefix_not_final_score=natural is None,full_fivefold_complete=False)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','actual_utc','actual_completed_epoch','actual_step_prefix','natural_exit','stderr_has_Traceback','last_FIT_step')}),flush=True)

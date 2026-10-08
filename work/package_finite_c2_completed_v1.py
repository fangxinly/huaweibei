from pathlib import Path
import argparse,json,hashlib,tarfile,datetime,subprocess,shutil
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sel=json.loads((r/'run/selection.json').read_text());h=json.loads((r/'run/history.json').read_text());exit=json.loads((r/'exit.json').read_text());pr=json.loads((r/'run/protocol.json').read_text());plan=json.loads((r/'finite_formal_plan_v2.json').read_text())
assert exit['exit_code']==0 and len(h)==sel['epochs']==100
assert sel['best_epoch']==min(range(1,101),key=lambda i:h[i-1]['dev_author_batch_mean_mse'])
assert sha(r/'run/best_addon.pt')==sel['addon_sha256'] and sha(r/'run/predictions.npz')==sel['prediction_sha256'] and sha(r/'run/shared_phase_addon.pt')==sel['shared_phase_sha256']
assert sha(r/'finite_formal_plan_v2.json')==pr['formal_plan_sha256']
for n,v in plan['source_sha256'].items():assert sha(r/n)==v
out=r/'completed_package';out.mkdir(exist_ok=False)
names=list(plan['source_sha256'])+['finite_formal_plan_v2.json','train_scales.json','head_fit_rows.npy','head_heldout_rows.npy','launch.json','exit.json']+['run/'+n for n in ['protocol.json','history.json','selection.json','predictions.npz','orders.npy','best_addon.pt','shared_phase_addon.pt']]
meta={n:dict(sha256=sha(r/n),bytes=(r/n).stat().st_size) for n in names}
pending=out/'completed.pending';done=out/'completed.tar.gz'
with tarfile.open(pending,'w:gz') as t:
 for n in names:t.add(r/n,arcname=n,recursive=False)
pending.replace(done)
receipt=dict(status='C2_ACTUAL_100_EXIT0_SELECTED_ADDON_PACKAGE_NOT_FULL_CHECKPOINT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection=sel,exit=exit,members=meta,package_sha256=sha(done),package_bytes=done.stat().st_size,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader'],text=True).strip(),compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip(),data_disk_free=shutil.disk_usage('/data').free,source_sha256=sha(__file__))
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('C2_COMPLETE_ADDON_PACKAGE_READY',sel['best_epoch'],flush=True)

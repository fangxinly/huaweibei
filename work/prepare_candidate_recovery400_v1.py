"""Recover interrupted fixed candidate at saved step400; retain algorithm/order/selection."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def prepare(tag):
 old=base/'work/polarity_intensity_resume_preparation_20261008T205030Z';new=base/'work'/('candidate_recovery400_preparation_'+tag);new.mkdir()
 for f in old.iterdir():
  if f.suffix in ('.py','.npy'):shutil.copy2(f,new/f.name)
 f=new/'resume_official_mse100_v1.py';s=f.read_text(encoding='utf8')
 s=s.replace("assert sha(a.resume)==p['prefix16_checkpoint_SHA']","assert p['recovery400_CPU_qualified'] and p['recovery_native_qualified'];assert sha(a.resume)==p['resume_checkpoint_SHA']")
 s=s.replace("assert pm['steps']==16 and pm['next_epoch']==0 and pm['next_batch']==16 and pm['not_selected_checkpoint']","assert pm['steps']==400 and pm['next_epoch']==10 and pm['next_batch']==0")
 s=s.replace('scheduler.last_epoch==16','scheduler.last_epoch==400').replace("float(q['step'])==16","float(q['step'])==400")
 marker="    write(a.out/'complete_prefix_restore.json'"
 start=s.index(marker);end=s.index('\n    del prefix,prefix_rng',start)
 s=s[:start]+"    restored_history=prefix['history'];assert len(restored_history)==10\n    restored_selected=cpu(prefix['selected_model']);restored_best=float(pm['best_MSE']);restored_bestepoch=int(pm['best_epoch'])\n    prior=Path(p['interrupted_prior_root'])/'out';bestfile=prior/('DEV_epoch_%03d_prediction_only.npz'%restored_bestepoch)\n    assert sha(bestfile)==restored_history[restored_bestepoch-1]['prediction_SHA'];restored_bestpred=np.load(bestfile,allow_pickle=False)['prediction'].copy()\n    assert tensor_sha(restored_selected)==restored_history[restored_bestepoch-1]['state_SHA']\n    for h in restored_history:\n        name='DEV_epoch_%03d_prediction_only.npz'%h['epoch'];assert sha(prior/name)==h['prediction_SHA'];shutil.copy2(prior/name,a.out/name)\n    shutil.copy2(prior/'DEV_selection_targets.npy',a.out/'DEV_selection_targets.npy')\n    write(a.out/'complete_recovery400_restore.json',dict(actual_UTC=utc(),input_SHA=p['resume_checkpoint_SHA'],restored_state_SHA=pm['final_state_SHA'],restored_optimizer_steps=400,scheduler_step=400,start_next_epoch=10,first400_counted_in4000=True,full_RNG_restored=True,prior_interrupted_training_natural_exit_unknown=True,replayed_work_not_extra_independent_budget=True))"+s[end:]
 s=s.replace("history=[];best=float('inf');selected=None;bestepoch=0;bestpred=None;steps=16;",'history=restored_history;best=restored_best;selected=restored_selected;bestepoch=restored_bestepoch;bestpred=restored_bestpred;steps=400;')
 s=s.replace('for epoch in range(100):','for epoch in range(10,100):').replace('if epoch==0 and j<16: continue','if False: continue')
 s=s.replace("history.append(record);write(a.out/'progress.json',record);", "prior_record_path=prior/'history.json'\n        oldhistory=read(prior_record_path)\n        if epoch<len(oldhistory):\n            expected=oldhistory[epoch];assert expected['prediction_SHA']==frozenSHA and expected['state_SHA']==stateSHA,'Recovered epoch diverges from interrupted deterministic trajectory'\n        history.append(record);write(a.out/'progress.json',record);")
 s=s.replace('if steps==400:', 'if steps%400==0:')
 s=s.replace('steps=400,next_epoch=10,next_batch=0','steps=steps,next_epoch=epoch+1,next_batch=0')
 s=s.replace("warmcp=a.out/'complete_resume_step400.pt';torch.save(warm,warmcp);del warm", "warmtmp=a.out/'mutable_recovery_checkpoint.tmp';torch.save(warm,warmtmp);del warm\n            warmcp=a.out/'mutable_recovery_checkpoint.pt';os.replace(warmtmp,warmcp)")
 s=s.replace("a.out/'warmup400_preservation_ready.json'", "a.out/'mutable_recovery_checkpoint_ready.json'").replace('steps=400,full_model_Adam','steps=steps,full_model_Adam')
 s=s.replace("prefix16_checkpoint_SHA=p['prefix16_checkpoint_SHA'],prefix_updates_counted_in4000=True", "prefix16_checkpoint_SHA=p['prefix16_checkpoint_SHA'],recovery400_checkpoint_SHA=p['resume_checkpoint_SHA'],prefix_updates_counted_in4000=True,interrupted_original_preserved=True,recovery_replayed_updates_not_new_independent_replicate=True")
 assert 'steps=16;' not in s and 'range(10,100)' in s
 f.write_text(s,encoding='utf8');ast.parse(s)
 native=new/'qualify_resume_official_v1.py';t=native.read_text(encoding='utf8').replace('range(16):','range(400):').replace('range(16,40):','range(400,440):').replace('expected[i-16]','expected[i-400]').replace('rs.last_epoch==16','rs.last_epoch==400').replace('rs.last_epoch==40','rs.last_epoch==440').replace('PREFIX16_RESUME_TO40','RECOVERY400_RESUME_TO440').replace('final_steps=40','final_steps=440')
 t=t.replace('rs.last_epoch==4400','rs.last_epoch==400')
 native.write_text(t,encoding='utf8');ast.parse(t)
 wrapper=new/'run_resumed100_capture_v1.py';w=wrapper.read_text(encoding='utf8').replace("sha(a.resume)==p['prefix16_checkpoint_SHA']","sha(a.resume)==p['resume_checkpoint_SHA']");wrapper.write_text(w,encoding='utf8');ast.parse(w)
 # Source-only native plan: no labels/new training. New real recovery plan requires actual peer400 verification.
 plan=json.loads((old/'resume_native_preparation_plan.json').read_text());plan.update(status='CANDIDATE_RECOVERY400_NATIVE_PREPARATION',source_sha256={f.name:sha(f) for f in new.iterdir() if f.suffix in ('.py','.npy')},recovery400_CPU_qualified=False,recovery_native_qualified=False)
 pf=new/'resume_native_preparation_plan.json';pf.write_text(json.dumps(plan,indent=2),encoding='utf8')
 with zipfile.ZipFile(new/'resume_native_source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in new.iterdir():
   if f.suffix!='.zip':z.write(f,f.name)
 print(json.dumps(dict(root=str(new),plan_SHA=sha(pf),bundle_SHA=sha(new/'resume_native_source.zip'),real_recovery_started=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();prepare(a.tag)

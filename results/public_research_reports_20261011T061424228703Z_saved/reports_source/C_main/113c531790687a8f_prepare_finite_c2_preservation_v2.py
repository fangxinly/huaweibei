from pathlib import Path
import argparse,json,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--assembly',type=Path,required=True);a=p.parse_args();r=a.assembly;d=r/'full_checkpoints/c';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n in ['protocol.json','history.json','selection.json','predictions.npz','orders.npy','best_addon.pt','shared_phase_addon.pt']:shutil.copy2(r/'run'/n,d/n)
plan=json.loads((r/'finite_formal_plan_v2.json').read_text())
source=list(plan['source_sha256'])
for n in ['finite_formal_plan_v2.json','train_scales.json','assemble_finite_full_checkpoint_v2.py']+source:shutil.copy2(r/n,d/n)
meta=lambda names:{n:dict(sha256=sha(d/n),bytes=(d/n).stat().st_size) for n in names}
required=['full_checkpoint.pt','full_checkpoint_receipt.json','protocol.json','history.json','selection.json','predictions.npz','orders.npy']
extra=['best_addon.pt','shared_phase_addon.pt','full_replay_predictions.npz','finite_formal_plan_v2.json','train_scales.json','assemble_finite_full_checkpoint_v2.py']+source
m=dict(training_node='c',assembly_node='a',required_seven_files=meta(required),extra_files=meta(extra),full_checkpoint_receipt=json.loads((d/'full_checkpoint_receipt.json').read_text()),amendment_id=plan['amendment_id'])
(d/'preservation_manifest.json').write_text(json.dumps(m,indent=2));print('C2_SEVEN_PLUS',len(extra),'PRESERVATION_MANIFEST_COMPLETE')

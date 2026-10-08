from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('--deployment',type=Path,required=True);a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for node in ['a','b']:
 run=a.deployment/('run' if node=='a' else 'incoming_completed/b/run');d=a.deployment/'full_checkpoints'/node
 for n in ['protocol.json','history.json','selection.json','predictions.npz','orders.npy','best_addon.pt','shared_phase_addon.pt']:shutil.copy2(run/n,d/n)
 for n in ['finite_formal_plan_v1.json','train_scales.json','assemble_finite_full_checkpoint_v1.py']:shutil.copy2(a.deployment/n,d/n)
 r=json.loads((d/'full_checkpoint_receipt.json').read_text())
 meta=lambda names:{n:dict(sha256=sha(d/n),bytes=(d/n).stat().st_size) for n in names}
 m=dict(training_node=node,assembly_node='a',required_seven_files=meta(['full_checkpoint.pt','full_checkpoint_receipt.json','protocol.json','history.json','selection.json','predictions.npz','orders.npy']),extra_files=meta(['best_addon.pt','shared_phase_addon.pt','full_replay_predictions.npz','finite_formal_plan_v1.json','train_scales.json','assemble_finite_full_checkpoint_v1.py']),full_checkpoint_receipt=r)
 (d/'preservation_manifest.json').write_text(json.dumps(m,indent=2));print('FINITE_PRESERVATION_MANIFEST',node)

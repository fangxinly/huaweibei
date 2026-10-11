import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
base=Path('D:/CodexBackups/selective_flow_20261003_1105');train=base/'weak_retention_train40_20261007T145201Z'
joint=train/'training_D_B_CPU_joint.json';assert json.loads(joint.read_text(encoding='utf8'))['status']=='PILOT40_D_ORIGINAL_CPU_CAPTURE_COMPLETE'
stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';d=base/('fixed_weak_donor_mask_'+stamp);assert not d.exists();d.mkdir()
source=Path('work/fixed_weak_pilot_donor_mask_diagnostic.py');shutil.copy2(source,d/source.name);shutil.copy2(joint,d/joint.name)
remote='/data/coding/fixed_weak_donor_mask_'+stamp
plan=dict(status='FIXED_WEAK_SELECTED36_DONOR_MASK_EVALUATION_FROZEN',actualclock_freeze_UTC=a.clock,
 diagnostic_source_sha256=sha(source),parent_roots={'A':'/data/coding/weak_retention_train40_run_20261007T145201Z','B':'/data/coding/weak_retention_train40_B_original_20261007T145201Z'},
 A_bundle='/data/coding/weak_retention_train40_source_20261007T145201Z',B_bundle='/data/coding/weak_retention_train40_source_20261007T145201Z',
 parent_receipt_sha256=sha(train/'A_original/out/actual_stage_receipt.json'),checkpoint_sha256='e173bf8b56327161a7fff430597cf7441e6d51d4a35381fa14c3b8743a01d7e6',
 selected_state_sha256='ad9c35a4b7900dbd8806f11a2466bbdc771211f52c44d09729b725fd10245a24',
 preservation_joint_path={k:'/data/coding/weak_train40_preservation_joint_20261007T145201Z.json' for k in ('A','B')},preservation_joint_SHA=sha(joint),
 assets='/data/coding/multimodal_flow_public_20261007T141430Z',A_uuid='GPU-4e64fc9f-5481-c32d-3fd4-e16a3a9a7198',
 lease_end='2026-10-08T14:00:00+00:00',lease_basis='Human new batch 24h, conservative bound, platform not verified',gpu_seconds=900,
 same_checkpoint_fixed_selected_epoch=36,fit_inner_masks=['off','on','audio_to_text','vision_to_text','text_to_audio','vision_to_audio','text_to_vision','audio_to_vision'],
 mechanism='Multiply each donor feedback MLP output by fixed 0/1 hooks. Same own fields, gain, Euler, decoder, reader and role head. No source/weight updates; does not disable every crossmodal path.',
 cached_inference='Per batch use one encoded source for fixed masks; all-on cached result must exactly equal original full forward; dummy0/7 repeat all masks; full state/RNG/FIT normalization unchanged.',
 official_VAL_TEST='Only fixed p/b/f descriptive metrics. Pooled fit/inner/outer overlap disclosed; not retained TEST or full five-fold OOF. No counterfactual mask selection on TEST.',
 no_optimizer_updates=True,no_method_or_checkpoint_selection=True,D_physical_free_bytes=shutil.disk_usage(base).free,
 local_evidence_estimate_bytes=25000000,no_large_checkpoint_redownload=True)
assert plan['D_physical_free_bytes']>plan['local_evidence_estimate_bytes']+1000000000
write(d/'protocol.json',plan);shutil.copy2(__file__,d/Path(__file__).name)
with zipfile.ZipFile(d/'frozen_source_plan.zip','x',zipfile.ZIP_DEFLATED) as z:
 for item in d.iterdir():
  if item.is_file() and item.suffix!='.zip':z.write(item,item.name)
pointer=dict(D=str(d),remote=remote,source='/data/coding/fixed_weak_pilot_donor_mask_diagnostic_'+stamp+'.py',plan='/data/coding/fixed_weak_donor_mask_plan_'+stamp+'.json',plan_SHA=sha(d/'protocol.json'),source_SHA=sha(source),freeze_zip_SHA=sha(d/'frozen_source_plan.zip'))
write(Path('work/fixed_weak_donor_mask_current.json'),pointer);write(d/'freeze_receipt.json',pointer);print(json.dumps(pointer))

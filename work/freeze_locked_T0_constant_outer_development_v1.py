"""Metadata-only role and asset check; freezes before outer prediction/label values."""
import hashlib,json,zipfile,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/locked_T0_constant_outer_development_20261006T1324Z';OUT.mkdir(exist_ok=False)
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_completed_20261006/a')
ROLES=ROOT/'work/group_teacher_plan_20261005T1650Z'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:datetime.now(timezone.utc).isoformat()
def pin(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
manifest=json.loads((BASE/'preservation_manifest.json').read_text())
paths={'outer_predictions':BASE/'run_v1/outer_scalar_predictions.npz','teacher_completion':BASE/'run_v1/completion.json','teacher_manifest':BASE/'preservation_manifest.json',
 'mapping':BASE/'train_row_video_mapping.json','teacher_plan':ROLES/'group_teacher_plan_v3.json','fit_rows':ROLES/'fit_0.npy','inner_rows':ROLES/'inner_0.npy','outer_rows':ROLES/'outer_0.npy',
 'old_rule':ROOT/'work/plain_scalar_residual_fold0_20261006T1157Z/execute/A_frozen_rule.json','old_plan':ROOT/'work/plain_scalar_residual_fold0_20261006T1157Z/plan.json',
 'labels':Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_oof_20261006T0308Z/train_scalar_oof.npz')}
pins={k:pin(v) for k,v in paths.items()}
for name in ('outer_predictions','teacher_completion'):
 relative=paths[name].relative_to(BASE).as_posix();assert pins[name]['sha256']==manifest['required_seven_files'][relative]['sha256']
assert pins['labels']['sha256']=='cff1a52f7e828c96cacf070ae4e051a5c34042ebb9435f44f5b3242cad91a7ea'
assert pins['teacher_plan']['sha256']=='f0878166da09433c27119a08fdfd8682f9b9e29f2bfe0a758f29eee45cf131a4'
ids={role:np.load(paths[role+'_rows'],allow_pickle=False) for role in ('fit','inner','outer')}
mapping=json.loads(paths['mapping'].read_text());v=np.array([x['video_id'] for x in mapping])
assert len(mapping)==1281 and [x['row'] for x in mapping]==list(range(1281))
sets={r:set(v[a]) for r,a in ids.items()};assert [len(ids[r]) for r in ids]==[695,153,433] and [len(sets[r]) for r in ids]==[30,4,18]
assert not sets['fit']&sets['inner'] and not sets['fit']&sets['outer'] and not sets['inner']&sets['outer']
role_graph=[]
for vid in sorted(set(v)):
 role=next(r for r in sets if vid in sets[r])
 role_graph.append({'video':str(vid),'rows':int(np.sum(v==vid)),'T0_reference_role':role,'T0_fit_label_access':role=='fit','T0_checkpoint_selection_label_access':role=='inner',
  'old_constant_head_label_access':role=='fit','old_scalar_head_evaluation':role=='inner','old_all_TRAIN_exploration_exists':True,'fresh_confirmation_eligible':False,
  'proposed_measurement':'LOCKED_CONSTANT_DEVELOPMENT_ONLY' if role=='outer' else 'NOT_REEVALUATED'})
(OUT/'label_asset_access_graph.json').write_text(json.dumps({'actual_utc':utc(),'videos':role_graph,'OUTER_433_excluded_from_T0_fit_inner_and_head_fit':True,'old_all_TRAIN_history_not_erased':True,'no_fresh_confirmation_videos_claimed':True,'outer_label_or_prediction_values_decoded':False,'correct_original_completed_relative_path':'run_v1/completion.json; remote fold directory must be derived from original trainer before querying'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
headers={}
for key in ('outer_predictions','labels'):
 with zipfile.ZipFile(paths[key]) as z:
  fields={}
  for name in z.namelist():
   with z.open(name) as f:
    version=np.lib.format.read_magic(f);read=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    shape,fortran,dtype=read(f);fields[name]={'shape':list(shape),'dtype':str(dtype),'fortran':fortran}
  headers[key]=fields
(OUT/'header_only_asset_receipt.json').write_text(json.dumps({'actual_utc':utc(),'headers':headers,'prediction_or_label_values_decoded':False,'source_pins':pins},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
c=json.loads(paths['old_rule'].read_text())['constant'];assert c==-0.07411910583740507
plan={'scope':'ONE_LOCKED_T0_CONSTANT_OUTER_DEVELOPMENT','frozen_actual_utc':utc(),'source_sha256':sha(ROOT/'work/locked_T0_constant_outer_development_v1.py'),'fold':0,'checkpoint_sha256':'338df042d648d3bdcb221bc0cacbd3d5574a1dd1efac49b1823f091a75706695','locked_constant':c,'pins':pins,
 'comparisons':['Original T0 F','Original T0 F plus same previously frozen FIT-only constant'], 'rows':433,'videos':18,'no_refit_no_sweeps':True,
 'primary':'Video equal paired squared loss difference constant minus F','secondary':['Pooled MSE/MAE','every video Q','delete-one-video Q','improved video count'],
 'cost_heuristic':'meanQ<0 and at least12of18 videos improve and every delete-one meanQ<0; no statistical or safety claim',
 'stop':'No switching fold/seed/sign/scale/refit. Constant fails: stop extending this scalar branch. Pass: retain only development baseline, not authorization for message/Q/new100.',
 'labels':'Decode only exact T0 OUTER433 y after locked prediction SHA. Do not read FIT/INNER values or mixed OOF mu. Old CAL/EVAL row overlap and all TRAIN exploration recorded, not fresh confirmation.',
 'new_confirmation':False,'whole_pipeline_crossfit':False,'new_GPU_or_Torch':False,'minimum_D_free_bytes':1073741824,'maximum_seconds':60,'maximum_output_bytes':1048576,'asset_graph_sha256':sha(OUT/'label_asset_access_graph.json')}
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'HEADER_ROLE_ASSET_GRAPH_PASSED_PLAN_FROZEN_BEFORE_VALUES','directory':str(OUT),'plan_sha256':sha(OUT/'plan.json'),'outer_labels_or_prediction_values_decoded':False,'free_bytes':{p:shutil.disk_usage(p).free for p in ('C:/','D:/')}}))

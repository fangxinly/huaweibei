from pathlib import Path
import json,zipfile,io,hashlib,datetime
import numpy as np
w=Path(__file__).parent;repo=w.parent;o=repo/'outputs';out=w/'group_teacher_plan_20261005T1650Z';assert not out.exists();out.mkdir()
base=Path('D:/CodexBackups/selective_flow_20261003_1105');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with zipfile.ZipFile(base/'finite_c2_completed_snapshots_20261005T1627Z/a/snapshot.zip') as z:
 mapping=json.loads(z.read('train_group_mapping/train_row_video_mapping.json'));receipt=json.loads(z.read('train_group_mapping/receipt.json'));assert len(mapping)==1281
 groups={}
 for x in mapping:groups.setdefault(x['video_id'],[]).append(x['row'])
 folds=[]
 for k in range(3):
  outer=np.load(io.BytesIO(z.read(f'train_group_mapping/outer_hold_{k}.npy')));other=np.load(io.BytesIO(z.read(f'train_group_mapping/outer_other_{k}.npy')))
  candidates=sorted({mapping[int(i)]['video_id'] for i in other},key=lambda v:hashlib.sha256(('group_teacher_inner_v1:'+v).encode()).hexdigest())
  chosen=[];count=0;target=.15*len(other)
  for v in candidates:
   if count>=target:break
   chosen.append(v);count+=len(groups[v])
  inner=np.array(sorted(i for v in chosen for i in groups[v]),dtype=np.int64);fit=np.setdiff1d(other,inner)
  assert not set(fit)&set(inner) and not set(other)&set(outer) and set(fit)|set(inner)|set(outer)==set(range(1281))
  assert not {mapping[int(i)]['video_id'] for i in fit}&set(chosen)
  for label,value in [('fit',fit),('inner',inner),('outer',outer)]:np.save(out/f'{label}_{k}.npy',value)
  orders=np.stack([np.random.RandomState(91818+1327+e).permutation(fit) for e in range(100)]);np.save(out/f'orders_{k}.npy',orders)
  folds.append(dict(fold=k,node=['a','b','c'][k],fit_rows=len(fit),inner_rows=len(inner),outer_rows=len(outer),fit_videos=sorted({mapping[int(i)]['video_id'] for i in fit}),inner_videos=chosen,outer_videos=sorted({mapping[int(i)]['video_id'] for i in outer}),files={n:dict(sha256=sha(out/n),bytes=(out/n).stat().st_size) for n in [f'fit_{k}.npy',f'inner_{k}.npy',f'outer_{k}.npy',f'orders_{k}.npy']},updates_per_epoch=int(np.ceil(len(fit)/32)),total_updates=100*int(np.ceil(len(fit)/32))))
plan=dict(status='FROZEN_TRAIN_ONLY_GROUP_TEACHER_SPLITS_ARCHITECTURE_AND_MECHANISM_BUDGET_NOT_FORMAL_TRAINING',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=91818,epochs=100,folds=folds,architecture='PublicDeBERTaV3base + pinnedCaReFlow text/audio/vision encoders with explicitpadding/contentpooling and fit-only channel statistics + originalfusion/predictor. Remove allflows; scalarMSE teacher; all retained parameters finetuned. This is teacher material, not a replacement for directionalflow experiments.',loss='Only fit-row scalarMSE; no flow/utility/semantic auxiliary.',optimizer='Pinned author AdamW lr1e-5 weightdecay groups and10percent warmup, completefitorder includinglastpartialbatch.',selection='100epochs earlieststrictminimum INNER sample-weightedMSE, batch128; no DEV/TEST selection; originalouter labels never accessed before frozen100selected model.',scope='Only officialTRAIN1281. Picklecontainer containsother splits but noDEV/TEST entry indexed; tensorize onlyguardedfit/inner beforecompletion, outerinputs zero-label adapter aftercompletion.',normalization='fit_statistics onlyfit tensors. No inner/outer data, labels, normalization or pretrainedtaskcheckpoint. Trainable/non-normalization initialSHA matched acrossfolds; per-foldfit statistics buffers intentionallydiffer.',preflight_budget='Eachfold freshGPU4fitupdates, innerlabelreplacement, fit-onlystatisticswitness andguardedouteraccessdenial, strictinitialfull checkpointdiskreload/replay. Noformal100 until3originalreceipts/sources/wholeorders/budgetverified.',formal_budget_pending='Measure actualGPU times/memory anddisk beforecommit; reserve2hours forpreservation beforeestimatedleaseend. Atleast3newfullweights about2.24GB pluslocalarchive/independentCPU; user24h is estimate.',mapping_receipt=receipt)
(out/'group_teacher_plan_v1.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');(o/'视频分组任务教师划分与预检冻结计划.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(folds))

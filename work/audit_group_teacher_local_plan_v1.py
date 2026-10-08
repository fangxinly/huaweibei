from pathlib import Path
import json,zipfile,io,hashlib,ast,datetime
import numpy as np
w=Path(__file__).parent;root=w/'group_teacher_plan_20261005T1650Z'
plan=json.loads((root/'group_teacher_plan_v2.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with zipfile.ZipFile('D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_snapshots_20261005T1627Z/a/snapshot.zip') as z:
 mapping=json.loads(z.read('train_group_mapping/train_row_video_mapping.json'))
 assert len(mapping)==1281 and {x['row'] for x in mapping}==set(range(1281))
 folds=[];held=[]
 for f in plan['folds']:
  k=f['fold'];ids={role:np.load(root/f'{role}_{k}.npy') for role in ['fit','inner','outer']}
  assert np.array_equal(ids['outer'],np.load(io.BytesIO(z.read(f'train_group_mapping/outer_hold_{k}.npy'))))
  sets={role:set(rows.tolist()) for role,rows in ids.items()}
  videos={role:{mapping[int(i)]['video_id'] for i in rows} for role,rows in ids.items()}
  for x,y in [('fit','inner'),('fit','outer'),('inner','outer')]:assert not sets[x]&sets[y] and not videos[x]&videos[y]
  assert sets['fit']|sets['inner']|sets['outer']==set(range(1281))
  for role in ids:assert np.array_equal(ids[role],np.sort(ids[role])) and len(ids[role])==f[role+'_rows'] and videos[role]==set(f[role+'_videos'])
  orders=np.load(root/f'orders_{k}.npy');assert orders.shape==(100,len(ids['fit']))
  for e,row in enumerate(orders):
   assert np.array_equal(np.sort(row),ids['fit'])
   assert np.array_equal(row,np.random.RandomState(91818+1327+e).permutation(ids['fit']))
  for name,reference in f['files'].items():assert sha(root/name)==reference['sha256'] and (root/name).stat().st_size==reference['bytes']
  held.extend(ids['outer'].tolist());folds.append({'fold':k,'fit':len(ids['fit']),'inner':len(ids['inner']),'outer':len(ids['outer']),'video_disjoint':True,'all100orders_verified':True})
assert sorted(held)==list(range(1281))
for name,reference in plan['source_sha256'].items():
 assert sha(root/name)==reference==sha(w/name)
 ast.parse((root/name).read_text(encoding='utf-8'))
tree=ast.parse((root/'group_teacher_runtime_v1.py').read_text(encoding='utf-8'))
plain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plain_masked')
assert not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='label_ids' for n in ast.walk(plain))
assert not any(isinstance(n,ast.Constant) and n.value in ['dev','test'] for n in ast.walk(tree))
assert plan['epochs']==100 and plan['seed']==91818
out={'status':'LOCAL_PLAN_ARRAY_SOURCE_AND_AST_CHECKS_PASSED_NOT_GPU_MECHANISM_VALIDATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':sha(root/'group_teacher_plan_v2.json'),'folds':folds,'outer_full_coverage':True,'masked_forward_does_not_load_label_argument':True,'runtime_no_dev_or_test_split_literal':True,'source_sha256':plan['source_sha256'],'GPU_precheck_executed':False,'OOF_predictions_generated':False,'formal_training_started':False,'limitation':'AST inspection and NumPy arrays do not establish actual GPU gradients, initialization equivalence, guard execution, memory, or time budget. Public helper main contains DEV/TEST evaluation and must never be invoked by this teacher runtime.'}
target=w.parent/'outputs/视频分组任务教师本地冻结源与划分独立核验.json'
target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out))

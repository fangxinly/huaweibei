from pathlib import Path
import json,zipfile,io,hashlib,datetime
import numpy as np
w=Path(__file__).parent;o=w.parent/'outputs';d=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_snapshots_20261005T1627Z/a');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(d/'snapshot.zip') as z:
 r=json.loads(z.read('train_group_mapping/receipt.json'));rows=json.loads(z.read('train_group_mapping/train_row_video_mapping.json'))
 assert r['source_sha256']==sha(z.read('train_group_mapping_source/inspect_train_group_mapping_v1.py'))==sha((w/'inspect_train_group_mapping_v1.py').read_bytes())
 assert r['mapping_sha256']==sha(z.read('train_group_mapping/train_row_video_mapping.json')) and r['rows']==len(rows)==1281 and r['video_groups']==len(set(x['video_id'] for x in rows))==52
 assert not r['dev_requested'] and not r['test_requested'] and r['source_order_labels_match_train_cache'] and [x['row'] for x in rows]==list(range(1281))
 allhold=[];vgroups=[]
 for f in r['folds']:
  k=f['fold'];hb=z.read(f'train_group_mapping/outer_hold_{k}.npy');ob=z.read(f'train_group_mapping/outer_other_{k}.npy');assert sha(hb)==f['hold_sha256'] and sha(ob)==f['other_sha256']
  h=np.load(io.BytesIO(hb));v=np.load(io.BytesIO(ob));assert len(h)==f['hold_rows'] and len(v)==f['other_rows'] and not set(h)&set(v) and set(h)|set(v)==set(range(1281))
  groups=set(rows[i]['video_id'] for i in h);assert groups==set(f['hold_video_ids']) and not groups&set(rows[i]['video_id'] for i in v);allhold.extend(h.tolist());vgroups.append(groups)
 assert sorted(allhold)==list(range(1281)) and all(not x&y for i,x in enumerate(vgroups) for y in vgroups[i+1:])
out=dict(status='TRAIN_ONLY52_VIDEO1281_ROW_THREE_GROUP_SPLITS_INDEPENDENTLY_AUDITED_NOT_CROSSFIT_TEACHERS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=r,hold_counts=[x['hold_rows'] for x in r['folds']],scope='Prerequisite only; no teacher fitting or OOF prediction. Existing fullTRAINfit/DEVselected cache cannot be called whole-pipeline crossfit. No DEV or TEST row use.')
(o/'TRAIN视频组映射与三外折先决核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out['hold_counts'])

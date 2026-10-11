from pathlib import Path
import argparse,pickle,json,hashlib,re,datetime,subprocess
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not a.out.exists();a.out.mkdir()
train=pickle.load((a.root/'assets/mosi.pkl').open('rb'))['train'];assert len(train)==1281
rows=[];groups={};y=[]
for i,x in enumerate(train):
 assert len(x)==3 and isinstance(x[-1],str);identifier=x[-1];m=re.fullmatch(r'(.+)\[(\d+)\]',identifier);assert m
 video=m.group(1);rows.append(dict(row=i,segment_id=identifier,video_id=video,segment_index=int(m.group(2))));groups.setdefault(video,[]).append(i);y.append(float(np.asarray(x[1]).reshape(-1)[0]))
assert len(set(x['segment_id'] for x in rows))==1281
with np.load(a.root/'teacher_cache_v1/train_cache.npz') as c:assert np.array_equal(np.asarray(y,dtype=np.float32),c['y'])
foldgroups=[[],[],[]];counts=[0,0,0]
# No labels in assignment: descending group size, then fixed SHA of video ID.
for video,ids in sorted(groups.items(),key=lambda x:(-len(x[1]),hashlib.sha256(('train_group_v1:'+x[0]).encode()).hexdigest())):
 k=min(range(3),key=lambda i:(counts[i],i));foldgroups[k].append(video);counts[k]+=len(ids)
folds=[]
for k,videos in enumerate(foldgroups):
 held=np.array(sorted(i for v in videos for i in groups[v]),dtype=np.int64);fit=np.setdiff1d(np.arange(1281),held)
 assert not set(videos)&set(v for j,g in enumerate(foldgroups) if j!=k for v in g)
 np.save(a.out/f'outer_hold_{k}.npy',held);np.save(a.out/f'outer_other_{k}.npy',fit);folds.append(dict(fold=k,hold_rows=len(held),other_rows=len(fit),hold_video_ids=videos,hold_sha256=sha(a.out/f'outer_hold_{k}.npy'),other_sha256=sha(a.out/f'outer_other_{k}.npy')))
assert set(np.concatenate([np.load(a.out/f'outer_hold_{k}.npy') for k in range(3)]))==set(range(1281))
(a.out/'train_row_video_mapping.json').write_text(json.dumps(rows,indent=2))
r=dict(status='TRAIN_ONLY_1281_ROW_VIDEO_GROUP_MAPPING_AND_LABEL_FREE_THREE_OUTER_SPLITS_VERIFIED_NOT_OOF_TEACHERS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),dataset_sha256=sha(a.root/'assets/mosi.pkl'),rows=1281,video_groups=len(groups),source_order_labels_match_train_cache=True,mapping_sha256=sha(a.out/'train_row_video_mapping.json'),folds=folds,dev_requested=False,test_requested=False,gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),scope='Read-only TRAIN mapping and partition prerequisite, no teacher fit or OOF predictions. Existing cached teacher was FULLTRAINfit/DEVselected and may not be called whole-pipeline crossfit. Group mapping does not itself establish fit/normalization/selection isolation.')
(a.out/'receipt.json').write_text(json.dumps(r,indent=2));print('TRAIN_GROUP_MAPPING_COMPLETE',len(groups),counts,flush=True)

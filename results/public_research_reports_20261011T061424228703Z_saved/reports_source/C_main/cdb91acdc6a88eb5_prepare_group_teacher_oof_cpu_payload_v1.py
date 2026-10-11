from pathlib import Path
import json,datetime,hashlib,shutil,zipfile
base=Path(__file__).resolve().parent.parent;parent=Path('D:/CodexBackups/selective_flow_20261003_1105');assert shutil.disk_usage(parent).free>2*1024**3
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=parent/('group_teacher_oof_cpu_payload_'+stamp);dest.mkdir()
completed=parent/'group_teacher_completed_20261006';oof=parent/'group_teacher_oof_20261006T0308Z'
sources={'oof/train_scalar_oof.npz':oof/'train_scalar_oof.npz','oof/analysis.json':oof/'analysis.json','audit_group_teacher_oof_arrays_v1.py':base/'work/audit_group_teacher_oof_arrays_v1.py','completed/a/train_row_video_mapping.json':completed/'a/train_row_video_mapping.json'}
for f,node in enumerate(['a','b','c']):
 for n in [f'fit_{f}.npy',f'outer_{f}.npy','run_v1/outer_scalar_predictions.npz','run_v1/completion.json','cpu_preservation_receipt.json']:
  sources[f'completed/{node}/{n}']=completed/node/n
meta={}
with zipfile.ZipFile(dest/'payload.zip','w',zipfile.ZIP_STORED) as z:
 for n,source in sources.items():
  raw=source.read_bytes();z.writestr(n,raw);meta[n]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':meta,'sha256':hashlib.sha256((dest/'payload.zip').read_bytes()).hexdigest(),'scope':'OOF scalar arrays and original underlying outer arrays only. Completed full-model CPU copies have separate original receipts; this payload is not another fullweight transport or training.'}
(dest/'manifest.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({'directory':str(dest),'sha256':r['sha256'],'members':len(meta)}))

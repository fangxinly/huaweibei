from pathlib import Path
import hashlib,json,zipfile,numpy as np,base64,datetime
root=Path(__file__).resolve().parents[1];checks=root/'work/new_p4_checks'
archive=Path('D:/CodexBackups/selective_flow_20261003_1105/teacher_cache_bundle_v1_20261005T1246Z.zip')
receipt=json.loads((checks/'teacher_cache_bundle_receipt.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
dest=archive.parent/'soft_teacher_cache_20261005T1246Z';dest.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert all(x.startswith('teacher_cache_v1/') and '..' not in Path(x).parts for x in z.namelist())
 z.extractall(dest)
cache=dest/'teacher_cache_v1';r=json.loads((cache/'collection.json').read_text())
assert r['source_sha256']==sha(root/'work/collect_soft_teacher_v1.py')
assert r['base_model_tensor_sha_before']==r['base_model_tensor_sha_after']
assert not r['test_split_requested'] and not r['dev_label_used_for_gradient_target'] and not r['formal_training_started']
for name,m in r['files'].items():assert sha(cache/name)==m['sha256'] and (cache/name).stat().st_size==m['bytes']
for split,n in [('train',1281),('dev',229)]:
 with np.load(cache/(split+'_cache.npz'),allow_pickle=False) as a:
  assert np.array_equal(a['row_id'],np.arange(n)) and a['state'].shape==(n,3,50,100)
  assert all(np.isfinite(a[k]).all() for k in a.files)
  assert ('teacher_gradient' in a.files)==(split=='train')
  if split=='train':recomputed=np.sqrt(np.mean(a['teacher_gradient'].astype(np.float64)**2,axis=(0,2)))
assert np.allclose(np.load(cache/'train_gradient_rms.npy'),recomputed,rtol=1e-6)
assets={}
for node in 'abc':
 b=''.join((checks/(node+'_assets.b64')).read_text().split());report=json.loads(base64.b64decode(b))
 (checks/(node+'_assets.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');assets[node]=report
out={'status':'CACHE_ZIP_ALL_MEMBER_SHA_TRAIN_RMS_AND_DEV_TARGET_ABSENCE_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(archive),'archive_sha256':receipt['sha256'],'cache':str(cache),'teacher_collection':r,'asset_reports':assets,'limits':'Frozen full-TRAIN/DEV-selected teacher is in-sample mechanism reference; no independent validation or semantic truth.'}
(root/'outputs/新P4固定教师缓存独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(out['status'])

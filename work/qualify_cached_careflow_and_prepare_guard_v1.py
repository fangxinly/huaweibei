"""Local byte preservation and synthetic access guard preparation; no model/data loading."""
import argparse, ast, hashlib, json, shutil, zipfile
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

p=argparse.ArgumentParser();p.add_argument('--actual-utc',required=True);args=p.parse_args()
stamp=args.actual_utc.replace('-','').replace(':','').replace(' ','T').replace('UTC','').strip()+'Z'
base=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
old=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen')
backup=Path('D:/CodexBackups/selective_flow_20261003_1105')
local=base/'work'/('official_baseline_artifact_guard_preparation_'+stamp)
D=backup/local.name
assert not local.exists() and not D.exists()
assert shutil.disk_usage(base).free>6*1024**3 and shutil.disk_usage(backup).free>6*1024**3
local.mkdir();D.mkdir()
expected='1ec10f20abdfb007495e7d8c7dc33cea1469427fe4bc5db3dcabb4f1afbf4365'
original=old/'outputs/server_recovery_20261003_1530/rescued_careflow_seed128.pt'
selection=old/'outputs/repeat5_experiments/completed/careflow_seed128/selection.json'
protocol=selection.with_name('protocol.json')
s=json.loads(selection.read_text(encoding='utf-8'))
assert s['checkpoint_sha256']==expected and s['protocol_sha256']==digest(protocol)
before=original.stat();assert before.st_size==742279993 and digest(original)==expected
with zipfile.ZipFile(original) as z:
    names=z.namelist();assert len(names)==len(set(names)) and z.testzip() is None
    entries=[{'name':i.filename,'bytes':i.file_size,'CRC':i.CRC} for i in z.infolist()]
after=original.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
saved=D/'original_selected_best.pt';shutil.copy2(original,saved)
assert saved.stat().st_size==742279993 and digest(saved)==expected
with zipfile.ZipFile(saved) as z:assert z.testzip() is None and z.namelist()==names
for path in (selection,protocol):shutil.copy2(path,local/('original_'+path.name))
guard='''"""Preparation only. Caller supplies an approved split; this module cannot open datasets."""
ALLOWED_ROLES = frozenset(('train', 'dev'))
EXPECTED_ROWS = {'train': 1281, 'dev': 229}

def approved_split(mapping, role, row_ids):
    # Reject before invoking even one operation on a potentially sensitive mapping.
    if role not in ALLOWED_ROLES:
        raise PermissionError('Only train and dev are approved during preparation/training')
    rows = mapping[role]
    ids = tuple(row_ids)
    if len(rows) != EXPECTED_ROWS[role] or len(ids) != len(rows):
        raise ValueError('Official row count or ID alignment mismatch')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate row IDs')
    return rows, ids

def require_disjoint_ids(train_ids, dev_ids):
    if set(train_ids).intersection(dev_ids):
        raise ValueError('Train and dev IDs overlap')
'''
(local/'train_dev_access_guard_preparation.py').write_text(guard,encoding='utf-8')
ast.parse(guard);ns={};exec(compile(guard,'synthetic_guard','exec'),ns)
class Sentinel:
    def __getitem__(self,key):raise AssertionError('Forbidden role touched underlying data')
for role in ('test','TEST','headEVAL201',None):
    try:ns['approved_split'](Sentinel(),role,())
    except PermissionError:pass
    else:raise AssertionError('Forbidden role accepted')
for role,n in (('train',1281),('dev',229)):
    rows=tuple(range(n));got=ns['approved_split']({role:rows},role,tuple(f'{role}:{i}' for i in rows))
    assert got[0] is rows
try:ns['require_disjoint_ids'](('x',),('x',))
except ValueError:pass
else:raise AssertionError('Role overlap accepted')
audit={'status':'ORIGINAL_CACHED_BASELINE_BYTES_PRESERVED_GUARD_PREPARATION_NOT_FORMAL_BENCHMARK',
       'clock_start_utc':args.actual_utc,'original_C':str(original),'saved_D':str(saved),
       'bytes':saved.stat().st_size,'sha256':expected,'ZIP_CRC_passed':True,'unique_members':len(entries),
       'original_selection_sha256':digest(selection),'original_protocol_sha256':digest(protocol),
       'best_epoch':s['best_epoch'],'history_manifest':'backup_audit_partial_20261003T072727Z.json',
       'original_old_D_direct_path_currently_exists':(backup/'careflow_seed128.pt').is_file(),
       'scope':'Existing C original copied locally to D; no remote download, Torch load, tensor decoding, prediction/data arrays, labels, inference, retraining or new scores',
       'complete_tensor_model_replay_qualification':False,'formal_cache_reuse_approved':False,
       'synthetic_access_guard_passed':True,'formal_full_source_runtime_frozen':False,
       'remaining':['Original tensor/state metadata and full fresh replay under exact source require separate gated execution',
                    'Common official roles, public assets, order/tail/update and DEV selection must be frozen for both methods',
                    'A new common epoch-order generator changes old baseline sampler chronology; matching cached epoch orders is unverified',
                    'Single method must not be chosen by 201 readout rankings; historical baseline TEST access stays disclosed']}
write(local/'artifact_byte_audit.json',audit);write(local/'checkpoint_ZIP_member_metadata.json',entries)
doc=f'''# 正式比较原baseline整文件与角色守卫准备

实际clock开始 {args.actual_utc}。在已存manifest中定位指定seed128的原selected best（epoch72），原C整文件742279993字节的fresh完整SHA与selection精确一致，ZIP完整CRC及唯一成员通过。原D顶层同名路径当前不存在；这只描述该路径，不推断所有D归档缺失。原C字节已本地复制至 {saved} 并重新核完整SHA/CRC；未远程重下载、未重训。

这通过整文件字节保存门，尚未加载Torch/state张量或运行完整模型重放；不据此宣称正式缓存已可复用。原protocol、源码、历史一次TEST访问边界继续保留。正式共同订单/尾批/更新/DEV选模尚未冻结；新共同订单生成器不能冒与旧缓存实际订单一致。

新增train/dev-only角色守卫只用合成哨兵核验：禁止角色在接触底层mapping前拒绝，合法角色核官方行数与唯一ID，TRAIN/DEV重叠拒绝。没有打开pickle/NPZ/真实标签、导入模型或执行GPU。该守卫尚未接入完整正式runner，不能冒运行时端到端标签隔离。
'''
(local/'preparation.md').write_text(doc,encoding='utf-8')
shutil.copytree(local,D/'preparation')
shutil.copy2(__file__,D/Path(__file__).name)
records={str(x.relative_to(D)).replace('\\','/'):digest(x) for x in D.rglob('*') if x.is_file() and x!=saved}
write(D/'small_member_SHA.json',records)
with zipfile.ZipFile(D/'small_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in [*records,'small_member_SHA.json']:z.write(D/name,name)
with zipfile.ZipFile(D/'small_snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(records)+1
    for name,h in records.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
write(D/'preservation_receipt.json',{'clock_start_utc':args.actual_utc,'large_original_actual_copy':str(saved),
      'large_original_sha256':expected,'small_zip_sha256':digest(D/'small_snapshot.zip'),
      'small_zip_member_count':len(records)+1,'small_zip_excludes_large_original':True,
      'large_original_is_actual_saved_file_not_SHA_reference_only':True,'no_remote_capture':True})
for ext,src in (('md',local/'preparation.md'),('json',local/'artifact_byte_audit.json')):
    shutil.copy2(src,base/'outputs'/('正式CaReFlow原整模型与标签守卫准备接续.'+ext))
write(local/'D_saved_location.json',{'D':str(D),'receipt':str(D/'preservation_receipt.json')})
print(json.dumps({'D':str(D),'original_bytes':saved.stat().st_size,'original_SHA_CRC':True,
                  'source_guard_synthetic_only':True,'no_new_experiment':True},ensure_ascii=False))

"""Count saved original targets, no predictions/models or new TEST metrics."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
clock=sys.argv[1];base=Path('D:/CodexBackups/selective_flow_20261003_1105');o=Path('outputs')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
full=base/'paired_fulltrain100_complete_actual_20261007T030311Z';saved={};provenance={}
for method in ['minimal_fixed_F','careflow']:
 root=full/method/'a/run/out'
 for role,name in [('TRAIN','original_TRAIN_supervision.npy'),('VAL','original_DEV_selection_targets.npy')]:
  p=root/name;assert p.exists();y=np.load(p,allow_pickle=False).astype(float).reshape(-1);saved[method+'_'+role]=y
  provenance[method+'_'+role]=dict(path=str(p),sha256=sha(p),rows=len(y))
assert np.array_equal(saved['minimal_fixed_F_TRAIN'],saved['careflow_TRAIN']) and np.array_equal(saved['minimal_fixed_F_VAL'],saved['careflow_VAL'])
prior=base/'C2_authorized_VAL_TEST_20261007T123036Z/A_original_capsule';m=json.loads((prior/'manifest.json').read_text(encoding='utf8'))['member_sha256'];tp=prior/'original/test_targets.npy';assert sha(tp)==m['original/test_targets.npy'];test=np.load(tp,allow_pickle=False).astype(float).reshape(-1)
for n in ['minimal_fixed_F_TEST','careflow_TEST']:
 saved[n]=test;provenance[n]=dict(path=str(tp),sha256=sha(tp),rows=len(test),scope='Author public official TEST targets previously scored; identical source/version/row identity qualified in fixed official prediction pipeline')
pilot=base/'selected20_requested_VAL_TEST_20261007T111903Z/A_original_capsule';pm=json.loads((pilot/'manifest.json').read_text(encoding='utf8'))['member_sha256']
for role in ['fit','inner']:
 p=pilot/'original'/(role+'_evaluation_targets.npy');assert sha(p)==pm['original/'+p.name];n='pilot_'+role;saved[n]=np.load(p,allow_pickle=False).astype(float).reshape(-1);provenance[n]=dict(path=str(p),sha256=sha(p),rows=len(saved[n]))
def count(y):
 assert np.isfinite(y).all() and (np.abs(y)<=3).all();weak=np.abs(y)<=1;strong=~weak
 return dict(rows=len(y),weak_abs_le1=int(weak.sum()),weak_percent=float(100*weak.mean()),strong_abs_gt1=int(strong.sum()),strong_percent=float(100*strong.mean()),neutral_y0=int((y==0).sum()),weak_nonzero=int((weak&(y!=0)).sum()),negative=int((y<0).sum()),positive=int((y>0).sum()),mean_abs_label=float(np.abs(y).mean()),mean_squared_label=float((y*y).mean()))
counts={n:count(y) for n,y in saved.items()};counts['all_pooled']=count(np.concatenate([saved['careflow_TRAIN'],saved['careflow_VAL'],test]))
report=dict(status='SAVED_ORIGINAL_TARGET_COUNTS_COMPLETE_NO_MODEL_OR_NEW_TEST_SCORE',actualclock_UTC=clock,weak_definition='abs(y)<=1 includes neutral y0 and +/-1 boundary',strong_definition='abs(y)>1',counts=counts,inputs=provenance,both_official_TRAIN_VAL_identical_exact=True,not_paper_raw_SDK_coverage=True,no_resampling_to_change_TEST_distribution=True,no_prediction_scoring=True)
stamp=clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';d=base/('sentiment_role_counts_'+stamp);d.mkdir()
(o/'强弱情绪比例原标签实际统计.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
md=['强弱情绪比例：原标签实际统计','',f'记录 {clock}。弱=|y|≤1（包括真值0），强=|y|>1。双方固定官方比较TRAIN/VAL原目标数组完全一致。','', '| 比较数据角色 | 样本 | 弱情绪 | 弱占比 | 强情绪 | 强占比 | 中性y=0 |','|---|---:|---:|---:|---:|---:|---:|']
for n,label in [('careflow_TRAIN','双方官方TRAIN'),('careflow_VAL','双方官方VAL'),('careflow_TEST','双方官方TEST'),('pilot_fit','新pilot FIT'),('pilot_inner','新pilot INNER'),('all_pooled','合并全部数据')]:
 c=counts[n];md.append(f"| {label} | {c['rows']} | {c['weak_abs_le1']} | {c['weak_percent']:.2f}% | {c['strong_abs_gt1']} | {c['strong_percent']:.2f}% | {c['neutral_y0']} |")
md+=['','以上为实际运行作者公共MOSI资产及已保存标签数组的条数，不冒原始SDK或论文不同清洗版本。未重算任何TEST模型分数；只统计用户明确询问的已评分目标分布。新pilot与官方对比不同划分，不能把其原角色混合TEST作为保留测试。TRAIN固定drop_last每轮1条的实际暴露分布需另从原订单统计，不把1281条全库比例冒实际4000批曝光。']
(o/'强弱情绪比例原标签实际统计.md').write_text('\n'.join(md),encoding='utf8')
for p in [o/'强弱情绪比例原标签实际统计.json',o/'强弱情绪比例原标签实际统计.md',Path(__file__)]:
 import shutil;shutil.copy2(p,d/p.name)
(d/'manifest.json').write_text(json.dumps(dict(actualclock_UTC=clock,sha256={p.name:sha(p) for p in d.iterdir()}),indent=2),encoding='utf8')
with zipfile.ZipFile(d/'complete_statistics.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in d.iterdir():
  if p.name!='complete_statistics.zip':z.write(p,p.name)
with zipfile.ZipFile(d/'complete_statistics.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
print(json.dumps(dict(counts=counts,D=str(d),zip_SHA=sha(d/'complete_statistics.zip')),ensure_ascii=False))

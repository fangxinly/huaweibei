"""Diagnostic-only loss audit on frozen DEV arrays; no fitting an inference model."""
from pathlib import Path
import datetime,hashlib,io,json,zipfile
import numpy as np
from analyze_utility_completed_v1 import digest,NAMES
out=Path('outputs');rows=[]
def huber(z):
 a=np.abs(z);return np.where(a<1,.5*a*a,a-.5)
def optimum(t,w):
 lo=-1.;hi=1.
 for _ in range(80):
  mid=(lo+hi)/2;grad=float(np.sum(w*np.clip(mid-t,-1,1)))
  if grad>0:hi=mid
  else:lo=mid
 theta=(lo+hi)/2;return theta,float(np.sum(w*huber(theta-t)))
def balanced(g,batches):
 w=np.zeros_like(g,dtype=float)
 for ix in batches:
  vals=g[ix];masks=[vals>0,vals<0];masks=[m for m in masks if m.any()]
  if not masks:w[ix]=1/(len(batches)*len(ix))
  else:
   for m in masks:w[ix[m]]=1/(len(batches)*len(masks)*int(m.sum()))
 assert np.isclose(w.sum(),1);return w
for node,folder in [('a','counterfactual_live_202610050613Z'),('b','counterfactual_live_202610050608Z')]:
 audit=json.loads((out/('主任务反事实效用'+node.upper()+'29条件诊断核验.json')).read_text(encoding='utf-8'));assert audit['status']=='COUNTERFACTUAL_29_CONDITION_FROZEN_DEV_ARRAYS_AUDITED';r=audit['rows'][0];archive=Path('work')/folder/node/'snapshot.zip';assert digest(archive)==r['archive_sha256']
 with zipfile.ZipFile(archive) as z:
  raw=z.read('counterfactual_diagnostics_20261005T0604Z/run_'+r['mode']+'/predictions.npz');assert hashlib.sha256(raw).hexdigest()==r['diagnostic']['output_sha256']
  with np.load(io.BytesIO(raw),allow_pickle=False) as f:d={k:f[k] for k in f.files}
 y=d['valid_y'].astype(float);p=d['pred_forced_fixed'].astype(float);u=d['utility_default'].astype(float);assert p.shape==y.shape==(229,)
 assert np.allclose(p,d['reference_prediction_default'],atol=2e-5,rtol=2e-5)
 directions=[]
 for i,name in enumerate(NAMES):
  off=d['pred_fixed_disable_'+str(i)].astype(float);delta=off-p;g=np.square(off-y)-np.square(p-y);assert np.max(np.abs(g-delta*(2*(p-y)+delta)))<1e-10;t=np.tanh(g/.05)
  weights={'natural_229':np.full(229,1/229),'global_sign_balanced':balanced(g,[np.arange(229)]),'batch128_101_sign_balanced':balanced(g,[np.arange(128),np.arange(128,229)])};objectives={}
  for label,w in weights.items():
   theta,minimum=optimum(t,w);actual=float(np.sum(w*huber(u[:,i]-t)));zero=float(np.sum(w*huber(t)))
   objectives[label]={'constant_huber_minimizer':theta,'target_weighted_mean':float(np.sum(w*t)),'constant_minimum_loss':minimum,'zero_constant_loss':zero,'observed_u_loss':actual,'observed_u_minus_zero_loss':actual-zero,'observed_u_minus_constant_minimum':actual-minimum,'constant_sign_matches_raw_mean_gain':bool(np.sign(theta)==np.sign(g.mean()))}
  directions.append({'direction':name,'raw_gain_mean':float(g.mean()),'raw_gain_std':float(g.std()),'positive_count':int((g>0).sum()),'negative_count':int((g<0).sum()),'exact_zero_count':int((g==0).sum()),'compressed_target_mean':float(t.mean()),'compressed_target_std':float(t.std()),'compression_absolute_saturation_fraction':float((np.abs(t)>.95).mean()),'observed_u_mean':float(u[:,i].mean()),'observed_u_std':float(u[:,i].std()),'raw_vs_compressed_mean_sign_reversal':bool(np.sign(g.mean())!=np.sign(t.mean())),'objectives':objectives})
 rows.append({'node':node,'mode':r['mode'],'archive_sha256':r['archive_sha256'],'arrays_sha256':r['diagnostic']['output_sha256'],'checkpoint_sha256':r['diagnostic']['checkpoint_sha256'],'directions':directions})
report={'status':'FIXED_REFERENCE_DEV_TARGET_TRANSFORM_AND_BALANCED_LOSS_DIAGNOSTICS_VERIFIED','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'auditor_sha256':digest(__file__),'limits':['DEV diagnosis only; no TRAIN teacher arrays available here and no inference model fitted.','Original tau=.05 retained; no DEV threshold or hyperparameter search.','Constants minimize a diagnostic objective over DEV labels and cannot be deployed or called an out-of-sample predictor.','Actual TRAIN uses evolving cached states, batches and teacher weights; these fixed DEV objectives do not identify the cause of training failure.','Only A/B available; C final missing.']}
with (out/'效用目标变换与均衡损失诊断.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# 已冻结DEV上的目标变换与均衡损失诊断','','仅复用A/B已核验的29条件数组，原τ=.05保持。以下常数只作为当前DEV目标的损失基准，没有训练或部署推断模型，不能用于DEV选结构或替代TRAIN外验证。C最终材料仍缺。','','|臂/方向|原始g均值|压缩t均值|全局正负均衡常数最优值|128/101批次均衡常数最优值|已学u均值|','|---|---:|---:|---:|---:|---:|']
for row in rows:
 for v in row['directions']:
  o=v['objectives'];lines.append(f"|{row['mode']} / {v['direction']}|{v['raw_gain_mean']:+.7f}|{v['compressed_target_mean']:+.7f}|{o['global_sign_balanced']['constant_huber_minimizer']:+.7f}|{o['batch128_101_sign_balanced']['constant_huber_minimizer']:+.7f}|{v['observed_u_mean']:+.7f}|")
reversals=[row['mode']+'/'+v['direction'] for row in rows for v in row['directions'] if v['raw_vs_compressed_mean_sign_reversal']]
imbalances=[row['mode']+'/'+v['direction'] for row in rows for v in row['directions'] if not v['objectives']['batch128_101_sign_balanced']['constant_sign_matches_raw_mean_gain']]
lines+=['',f"原始均值与压缩均值的符号不同：{', '.join(reversals) if reversals else '这12组中未观察到'}。这不撤销此前数学反例，只说明它并未在这些总体均值中出现。",'',f"按原128/101批次正负均衡后的诊断常数最优符号与原始风险均值不同：{', '.join(imbalances) if imbalances else '这12组中未观察到'}。数学上改变监督测度可以改变总体方向，但不能证明真实TRAIN失败由它造成。",'', 'B六方向已学u在128/101批次均衡诊断目标下的损失全部高于u=0基准；对应差值约.00018–.00054。它们的已学u均值为正，而原始g、压缩t和均衡常数最优值均为负。当前证据不能仅用总体目标符号反转解释这一偏差。也不能据此区分教师漂移、TRAIN/DEV分布差异、优化或特征不足；需要TRAIN教师数组与固定教师独立拟合诊断。', '', '每方向的正负样本数、压缩饱和比例、三种固定诊断目标、最优常数/零常数损失及已学u损失均保存于JSON。SmoothL1常数最优值用分段线性导数独立求根，包含非二次区间，不把加权均值无条件当精确最优值。', '', '原始g在当前fixed参考下反映终端平方风险。其条件均值、压缩目标的条件均值、正负均衡目标最优函数是三个不同对象。推断u不读取标签；这些DEV常数读取了DEV诊断标签，不能与可部署的逐样本预测器等同。后续若比较原风险目标，须先在TRAIN内冻结教师和监督测度，再作标签隔离与实际GPU检查。']
(out/'效用目标变换与均衡损失诊断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'raw_compressed_reversals':reversals,'batch_balancing_sign_mismatches':imbalances}))

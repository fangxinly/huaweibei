"""One fixed-checkpoint FIT/INNER comparison; no fitting, selection or TEST optimization."""
import sys,json,os,csv,zipfile,hashlib,datetime,shutil
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent.parent;ref=json.loads((root/'work/anchored_increment_pointer.json').read_text());s=Path(ref['source']);d=Path(ref['D']);sys.path.insert(0,str(s))
from common import sha,read,write
from sentiment_metrics_careflow_v1 import metrics
clock=sys.argv[1];stamp=sys.argv[2]
refs={k:read(d/k/'actual_D_verification.json') for k in ('A_native','B_native','A_cache','A_train','B_audit')};assert all(v['natural_exit']==0 for v in refs.values())
b=read(d/'B_audit/extracted/out/CPU_replay_result.json');assert b['status']=='ORIGINAL_TAIL_NEW_MESSAGE_SAVED_STATE_CPU_REPLAY_COMPLETE';assert max(v for r in b['errors'].values() for v in r.values())<=1e-4
tr=read(d/'A_train/extracted/out/train_result.json');predpath=d/'A_train/extracted/out/fixed_final_predictions.npz';assert sha(predpath)==tr['prediction_SHA'];trainplan=read(s/'increment_train_protocol.json')
out=root/'outputs'/('anchored_increment_review_'+stamp);out.mkdir();plan=dict(actualclock_score_freeze_UTC=clock,source_SHA=sha(__file__),train_plan_SHA=sha(s/'increment_train_protocol.json'),fixed_prediction_SHA=sha(predpath),metric_source_SHA=sha(s/'sentiment_metrics_careflow_v1.py'),source_and_prediction_D_CPU_refs=refs,metric_names=['Acc7','Acc2','F1','MAE','Corr'],weak_rule='abs(y)<=1',bootstrap_repeats=10000,bootstrap_seed=128,bootstrap_unit='INNER video, paired; concatenated row metrics, whole videos resampled',role_limitations='Explored INNER11 videos; original official TEST merged into FIT/INNER; not independent official benchmark; not full5fold',no_selection_or_fitting=True)
write(out/'score_protocol.json',plan);fd=os.open(d/'new_fixed_score_once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,sha(out/'score_protocol.json').encode());os.close(fd)
z=dict(np.load(predpath,allow_pickle=False));result=dict(actualclock=clock,actual_local_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),train_result=tr,CPU_result=b,protocol_SHA=sha(out/'score_protocol.json'),roles={})
for role in ('fit','inner'):
    records=list(csv.DictReader((s/('best36_'+role+'.csv')).open(encoding='utf8')));ids=[r['row_id'] for r in records];assert ids==z[role+'_ids'].tolist();y=np.asarray([float(r['y']) for r in records]);videos=np.asarray([r['video'] for r in records]);p0=z[role+'_p0'].astype(float);old=np.asarray([float(r['p1']) for r in records]);new=z[role+'_prediction'].astype(float);assert np.max(abs(p0-np.asarray([float(r['p0']) for r in records])))<=1e-5
    if role=='fit':
        yl=dict(np.load(d/'A_train/extracted/out/FIT_training_labels.npz',allow_pickle=False));assert np.array_equal(yl['y'].astype(float),y)
    table={name:{subset:metrics(v[mask],y[mask]) for subset,mask in [('overall',np.ones(len(y),dtype=bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]} for name,v in [('p0_same_flow_off',p0),('old_best36_message_on',old),('new_fixed20_message_on',new)]}
    rho=y-p0;delta=new-p0;X=np.column_stack([np.ones(len(y)),p0]);dhat=X@np.linalg.lstsq(X,delta,rcond=None)[0];rhat=X@np.linalg.lstsq(X,rho,rcond=None)[0]
    r2=1-float(np.sum((delta-dhat)**2)/np.sum((delta-delta.mean())**2));partial=float(np.corrcoef(delta-dhat,rho-rhat)[0,1]);utility=2*rho*delta-delta**2
    byvideo={v:dict(rows=int((videos==v).sum()),MSE_gain_p0_to_new=float(utility[videos==v].mean()),MAE_new_minus_p0=float((abs(new-y)-abs(p0-y))[videos==v].mean())) for v in sorted(set(videos))}
    r=dict(rows=len(y),videos=len(set(videos)),metrics=table,weak_rows=int((abs(y)<=1).sum()),strong_rows=int((abs(y)>1).sum()),delta_distribution=dict(mean=float(delta.mean()),std=float(delta.std()),min=float(delta.min()),max=float(delta.max())),new_delta_linear_p0_explained_R2=r2,new_delta_residual_partial_corr=partial,sample_squared_error_utility_positive_fraction=float((utility>0).mean()),per_video=byvideo)
    if role=='inner':
        vs=sorted(set(videos));indices=[np.flatnonzero(videos==v) for v in vs];rng=np.random.default_rng(128);names=['Acc7','Acc2','F1','MAE','Corr'];boot={baseline:[] for baseline in ('p0','old_best36')}
        for _ in range(10000):
            ix=np.concatenate([indices[j] for j in rng.integers(0,len(vs),len(vs))]);nm=metrics(new[ix],y[ix])
            for baseline,v in [('p0',p0),('old_best36',old)]:
                bm=metrics(v[ix],y[ix]);boot[baseline].append([nm[k]-bm[k] for k in names])
        r['paired_video_bootstrap_new_minus_baseline']={baseline:{k:dict(point=table['new_fixed20_message_on']['overall'][k]-table['p0_same_flow_off' if baseline=='p0' else 'old_best36_message_on']['overall'][k],CI95=np.quantile(np.asarray(vals)[:,i],[.025,.975]).tolist()) for i,k in enumerate(names)} for baseline,vals in boot.items()}
    result['roles'][role]=r
    with (out/(role+'_new_original_increment.csv')).open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['row_id','video','y','p0','old_p1','new_p1','new_delta'])
        for i in range(len(y)):w.writerow([ids[i],videos[i],y[i],p0[i],old[i],new[i],delta[i]])
write(out/'actual_five_metric_result.json',result)
lines=['# 原 AnchoredFlow 消息分支增量实验实测','',f'记录时间：{clock}。保留原 best36 编码器、100维状态、两步流、reader、解码器及 gain，仅六个400→8→100消息适配器训练20轮/940更新，共28,848个可训练参数。','', '消息为 h(接收方,供体)−h(接收方,零供体)，仍以0.125注入原第二步流。固定Huberδ1 +0.01 context²；FIT1494标签训练，INNER264仅固定最后模型评价，无INNER选epoch。','']
for role,r in result['roles'].items():
    lines += [f'## {role.upper()}：{r["rows"]}条/{r["videos"]}视频','', '| 模型 | Acc7 % | Acc2 % | F1 % | MAE | Corr | 弱MAE | 强MAE |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,t in r['metrics'].items():
        a=t['overall'];lines.append(f'| {name} | {100*a["Acc7"]:.4f} | {100*a["Acc2"]:.4f} | {100*a["F1"]:.4f} | {a["MAE"]:.6f} | {a["Corr"]:.6f} | {t["weak"]["MAE"]:.6f} | {t["strong"]["MAE"]:.6f} |')
    lines+=['',f'新Δ被p0线性函数解释R²={r["new_delta_linear_p0_explained_R2"]:.6f}；去除p0后的残差偏相关={r["new_delta_residual_partial_corr"]:.6f}。','']
ir=result['roles']['inner'];lines+=['## 固定 INNER 配对区间','', '| 比较（新−旧） | MAE 点差 | 视频bootstrap95%区间 |','|---|---:|---|']
for name,values in ir['paired_video_bootstrap_new_minus_baseline'].items():
    a=values['MAE'];lines.append(f'| {name} | {a["point"]:+.6f} | [{a["CI95"][0]:+.6f}, {a["CI95"][1]:+.6f}] |')
lines+=['','## 可作出的结论与限制','','这是原骨干上的小容量消息增量实验，不是上一轮公共冻结特征原型，也不是完整折外残差/效用控制器。原编码器已经使用FIT监督，不能把消息适配器的局部分折当作whole-pipeline OOF。','', 'INNER曾用于原模型选模，只有11个视频；合并训练包含原官方TEST记录。这里没有新的独立官方TEST结论、CaReFlow同预算比较或完整五折结果。所有五项来自同一固定最终检查点，不拼指标、不按本结果重挑超参。','', '完整新消息参数、Adam两矩、940步、20订单和RNG已保存；原骨干以完整parent SHA继承，可精确组合重建。这是组合状态，不能冒作新的独立完整整模型文件。A真实原生/缓存/训练与B原生/CPU尾部重放均自然0；B未做完整编码器CPU前向。','']
(out/'原方案消息增量改进实际报告.md').write_text('\n'.join(lines),encoding='utf8');shutil.copy2(__file__,out/Path(__file__).name)
manifest={f.name:sha(f) for f in out.iterdir()};write(out/'member_SHA.json',manifest);zp=d/'fixed20_result_report.zip'
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as zz:
    for f in out.iterdir():zz.write(f,f.name)
with zipfile.ZipFile(zp) as zz:
    assert zz.testzip() is None and len(zz.namelist())==len(set(zz.namelist()))
    for n,h in manifest.items():assert hashlib.sha256(zz.read(n)).hexdigest()==h
delivery=dict(actualclock=clock,report=str(out/'原方案消息增量改进实际报告.md'),result=str(out/'actual_five_metric_result.json'),D=str(d),report_ZIP=str(zp),report_SHA=sha(zp),SHA_CRC_unique=True,whole_goal_complete=False)
write(d/'fixed20_report_delivery.json',delivery);write(root/'outputs/原AnchoredFlow消息增量优化实际接续.json',delivery);print(json.dumps({'delivery':delivery,'INNER':ir},ensure_ascii=False))

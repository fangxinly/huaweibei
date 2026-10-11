"""Independent arithmetic and video-paired intervals for a fixed official run."""
import argparse, csv, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np

WS = Path(__file__).resolve().parent.parent
BASE = Path('D:/CodexBackups/selective_flow_20261003_1105')
KEYS = ('Acc7', 'Acc2', 'F1', 'MAE', 'Corr')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def independent(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    mask = y != 0
    truth, pred = y[mask] >= 0, p[mask] >= 0
    matrix = np.zeros((2, 2), float)
    np.add.at(matrix, (truth.astype(int), pred.astype(int)), 1)
    support = matrix.sum(1)
    denom = support + matrix.sum(0)
    f1 = np.divide(2 * matrix.diagonal(), denom, out=np.zeros(2), where=denom != 0)
    yc, pc = y - y.mean(), p - p.mean()
    corr = (yc @ pc) / np.sqrt((yc @ yc) * (pc @ pc))
    return dict(Acc7=float(np.mean(np.rint(np.clip(p, -3, 3)) == np.rint(np.clip(y, -3, 3)))),
                Acc2=float(np.trace(matrix) / matrix.sum()),
                F1=float(f1 @ support / support.sum()),
                MAE=float(np.abs(p - y).sum() / len(y)), Corr=float(corr))

def baseline(role):
    if role == 'TEST':
        path = BASE / 'paired_final_TEST_prediction_actual_20261007T043535Z/score_pair/run/out/joint_targets_and_fixed_predictions.npz'
        z = np.load(path, allow_pickle=False)
        return z['row_ids'], z['labels'].astype(float), z['careflow'], [path]
    path = BASE / 'paired_fulltrain100_complete_actual_20261007T030311Z/careflow/a/run/out/selected_best_DEV_replay.npz'
    labels = path.parent / 'original_DEV_selection_targets.npy'
    z = np.load(path, allow_pickle=False)
    return z['row_ids'], np.load(labels, allow_pickle=False).astype(float), z['prediction'], [path, labels]

def cluster_stats(p, y, video, unique):
    p, y = np.asarray(p, float), np.asarray(y, float)
    nonzero = y != 0
    truth, pred = y >= 0, p >= 0
    columns = np.column_stack([np.ones(len(y)), y, y*y, p, p*p, p*y, abs(p-y),
        np.rint(np.clip(p,-3,3)) == np.rint(np.clip(y,-3,3)),
        nonzero & ~truth & ~pred, nonzero & ~truth & pred,
        nonzero & truth & ~pred, nonzero & truth & pred]).astype(float)
    return np.stack([columns[video == v].sum(0) for v in unique])

def weighted_metrics(s):
    n, sy, sy2, sp, sp2, spy, ae, a7, tn, fp, fn, tp = s.T
    n0, n1 = tn+fp, fn+tp
    nz = n0+n1
    f0 = np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(n),where=2*tn+fp+fn!=0)
    f1 = np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(n),where=2*tp+fp+fn!=0)
    corr = (spy-sp*sy/n)/np.sqrt(np.maximum(0,sp2-sp*sp/n)*np.maximum(0,sy2-sy*sy/n))
    return np.column_stack([a7/n,(tp+tn)/nz,(f0*n0+f1*n1)/nz,ae/n,corr])

def paired(p, q, y, ids):
    video = np.asarray([str(s).split('[')[0] for s in ids])
    unique = np.unique(video)
    rng = np.random.default_rng(128)
    draws = rng.integers(len(unique), size=(10000,len(unique)))
    counts = np.zeros((10000,len(unique)),float)
    np.add.at(counts,(np.repeat(np.arange(10000),len(unique)),draws.reshape(-1)),1)
    d = weighted_metrics(counts @ cluster_stats(p,y,video,unique)) - weighted_metrics(counts @ cluster_stats(q,y,video,unique))
    pointp, pointq = independent(p,y), independent(q,y)
    assert np.allclose(weighted_metrics(cluster_stats(p,y,video,unique).sum(0)[None])[0],list(pointp.values()),atol=1e-12)
    return dict(videos=len(unique),seed=128,resamples=10000,unit='whole video paired resampling; row-weighted metrics',
                differences={k:dict(point=pointp[k]-pointq[k],percentile95=np.nanpercentile(d[:,i],[2.5,97.5]).tolist()) for i,k in enumerate(KEYS)},
                individual_intervals_not_joint_five_metric_significance=True)

def main(a):
    pointer = read(WS/'work/official_upgrade_pointer.json')
    plan = read(pointer['active_training_plan'])
    verified = {}
    for role in ('VAL','TEST'):
        ids,y,p,files = baseline(role)
        assert ids.tolist()==plan['official_role_IDs'][role]
        values = independent(p,y)
        err=max(abs(values[k]-plan['fixed_CaReFlow_five'][role][k]) for k in KEYS)
        assert err<1e-12
        verified[role]=dict(files={str(f):sha(f) for f in files},exact_canonical_ID_order=True,independent_metrics=values,error=err)
    if a.verify_baseline:
        print(json.dumps(verified)); return
    score_root = Path(pointer['D'])/'B_score/extracted/out'
    actual = read(score_root/'official_aligned_five_result.json')
    assert actual['status']=='OFFICIAL_TRAIN_VAL_TEST_ALIGNED_UPGRADE_CAREFLOW_FIVE_COMPLETE'
    frozen_path = Path(pointer['D'])/'A_infer/extracted/out/fixed_official_VAL_TEST_prediction.npz'
    assert sha(frozen_path) == actual['prediction_SHA']
    frozen = np.load(frozen_path, allow_pickle=False)
    training = read(Path(pointer['D'])/'A_train/extracted_small/out/training_result.json')
    audit = read(Path(pointer['D'])/'B_audit/extracted/out/audit_result.json')
    roles={}
    for role in ('VAL','TEST'):
        with (score_root/(role+'_actual_predictions.csv')).open(encoding='utf8',newline='') as f:
            rows=list(csv.DictReader(f))
        ids=np.asarray([r['row_id'] for r in rows]);y=np.asarray([float(r['y']) for r in rows])
        csvnew=np.asarray([float(r['new']) for r in rows]);csvoff=np.asarray([float(r['messages_off']) for r in rows])
        assert ids.tolist()==frozen[role+'_ids'].tolist()
        assert np.array_equal(csvnew.astype(np.float32),frozen[role+'_prediction'])
        assert np.array_equal(csvoff.astype(np.float32),frozen[role+'_p0'])
        new=frozen[role+'_prediction'].astype(float);off=frozen[role+'_p0'].astype(float)
        bid,by,b,_=baseline(role);assert ids.tolist()==bid.tolist() and np.array_equal(y,by)
        nv,ov=independent(new,y),independent(off,y)
        error=max(abs(values[k]-actual['roles'][role][key][k]) for values,key in ((nv,'new'),(ov,'messages_off')) for k in KEYS)
        assert error<1e-12
        groups={}
        for name,mask in [('weak',abs(y)<=1),('strong',abs(y)>1)]:
            groups[name]=dict(rows=int(mask.sum()),fraction=float(mask.mean()),new=independent(new[mask],y[mask]),messages_off=independent(off[mask],y[mask]),careflow=independent(b[mask],y[mask]))
        roles[role]=dict(actual=actual['roles'][role],independent_arithmetic_maxerror=error,
          paired_vs_careflow=paired(new,b,y,ids),paired_vs_messages_off=paired(new,off,y,ids),groups=groups)
    stamp=a.actualclock.replace('-','').replace(':','').replace(' UTC','Z').replace(' ','T')
    output=WS/'outputs'/('official_aligned_upgrade_review_'+stamp);output.mkdir()
    summary=dict(actualclock_UTC=a.actualclock,status='FIXED_OFFICIAL_TRAIN1281_VAL229_TEST685_COMPARISON_COMPLETE',roles=roles,
        training=training,other_node_CPU=audit,baseline_original_bytes=verified,
        frozen_prediction_source=dict(path=str(frozen_path),SHA=sha(frozen_path),CSV_float32_roundtrip_exact=True),
        reporting_operator_repair=read(Path('D:/CodexBackups/selective_flow_20261003_1105/official_anchored_upgrade_actual_20261008T053429Z/report_operator_failure_20261008T111806Z/actual_failure.json')),
        complete_parts_D_C=read(Path(pointer['D'])/'A_train/actual_split_D_C_verification.json'),
        source_pointer=pointer,whole_research_goal_complete=False,full_fivefold_complete=False,
        caveats=['Historical TEST was previously explored; this is a paired point comparison, not a fresh blinded benchmark.',
          'Same nominal updates/orders/selection; method losses, parameter counts and runtime differ.',
          'Messages-off is a functional ablation of the trained model, not a separately trained baseline.',
          'The upgrade does not yet implement the complete OOF residual-risk utility controller.'])
    result=output/'原方案改进与CaReFlow官方VAL_TEST实际结果.json'
    result.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    lines=[f'实际记录：{a.actualclock}。本次模型只使用官方 TRAIN1281 监督训练；VAL229 选模，固定后 TEST685 一次评价。',
      f'改进版100轮4000更新，VAL按批MSE严格最小、平局取最早，选第{training["metadata"]["best_epoch"]}轮。CaReFlow使用已完整保存并核验的第93轮。双方共享原100轮顺序、batch32/drop_last、seed128、VAL batch128/101 MSE选模规则及五指标定义。',
      '', '|划分|方法|Acc7 ↑|Acc2 ↑|F1 ↑|MAE ↓|Corr ↑|','|---|---|---:|---:|---:|---:|---:|']
    for role in ('VAL','TEST'):
        for label,key in [('CaReFlow','careflow'),('原方案改进','new'),('改进版同流关消息','messages_off')]:
            v=actual['roles'][role][key]
            lines.append(f'|{role}|{label}|{100*v["Acc7"]:.4f}%|{100*v["Acc2"]:.4f}%|{100*v["F1"]:.4f}%|{v["MAE"]:.9f}|{v["Corr"]:.9f}|')
    for role in ('VAL','TEST'):
        r=roles[role];ci=r['paired_vs_careflow']['differences']['MAE']['percentile95']
        lines.extend(['',f'{role}：五项同时严格点值超过CaReFlow={actual["roles"][role]["all_five_strict_point_improvement"]}。MAE差（改进版−CaReFlow）的按视频配对95%区间=[{ci[0]:+.6f}, {ci[1]:+.6f}]；全部五项区间见JSON，不将相关指标视作五个独立证据。'])
        for name in ('weak','strong'):
            g=r['groups'][name];lines.append(f'{name}（|y|{"≤" if name=="weak" else ">"}1）：{g["rows"]}条/{100*g["fraction"]:.3f}%；MAE改进/关消息/CaReFlow={g["new"]["MAE"]:.6f}/{g["messages_off"]["MAE"]:.6f}/{g["careflow"]["MAE"]:.6f}。')
    lines.extend(['','改动保留原100维、两步Euler、reader和共同读出。六方向低秩消息以供体相对零供体的增量注入，采用Huberδ1及固定context惩罚。所有组件从公共编码器和新任务初始化联合训练；没有加载合并数据检查点。此次仍是消息结构与损失改进，完整折外残差、校准和风险效用控制尚未实现。',
      '完整最终/最佳model、Adam4000、scheduler、RNG、100轮预测和原源码已以两份分片完整保存到D；C预留分片位置未使用，整ZIP字节SHA、CRC、唯一成员及源/argv/PID关联通过；B按原状态独立CPU验证并重放完整流消息，CPU误差='+str(audit['CPU_actual_DEV_flow_maxerror'])+'，不是CPU编码器前向。',
      '指标：Acc7对预测/标签clip[-3,3]后round；Acc2/F1剔除y=0、预测≥0判正，F1为support加权；MAE/Pearson使用原连续值。历史TEST已访问，不能称全新盲测或多种子稳定结论。此前合并模型的VAL .35449/TEST .36633含训练重叠，未纳入此比较；完整五折和五项超过的整体目标仍未完成。CSV默认float32十进制输出会引入极小舍入差，独立算术和bootstrap采用SHA核验的原始NPZ，CSV还原float32后逐值完全相同。'])
    report=output/'原方案改进与CaReFlow官方VAL_TEST总结报告.md';report.write_text('\n\n'.join(lines[:2])+'\n\n'+'\n'.join(lines[2:]),encoding='utf8')
    preserve=Path(pointer['D'])/'final_report';preserve.mkdir()
    for p in (report,result,Path(__file__)):shutil.copy2(p,preserve/p.name)
    members={p.name:sha(p) for p in preserve.iterdir() if p.is_file()}
    zpath=Path(pointer['D'])/'final_report_complete.zip'
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
        for n in members:z.write(preserve/n,n)
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    seal=dict(actualclock_UTC=a.actualclock,report=str(report),report_SHA=sha(report),JSON=str(result),JSON_SHA=sha(result),D=str(preserve),ZIP_SHA=sha(zpath),all_SHA_CRC_unique=True)
    (Path(pointer['D'])/'final_report_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(seal,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-baseline',action='store_true');p.add_argument('--actualclock');a=p.parse_args()
    assert a.verify_baseline or a.actualclock
    main(a)

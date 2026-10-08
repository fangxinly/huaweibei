"""Saved FIT/INNER arithmetic only. Never opens official VAL/TEST or checkpoints."""
import argparse, csv, datetime, hashlib, json, os, sys
from pathlib import Path
import numpy as np

EXPECTED = {
    'best20_fit_y_b_p_video.csv': 'ec4a06f7616d45a128fc8458192895b68d190a15adaa9ea4d046a56e77ba626c',
    'best20_inner_y_b_p_video.csv': 'b59e5535d8f54b011ab73f28416193008a3e148db5887703951a6eedfee02c0f',
    'best36_fit_y_b_p_video.csv': 'e80f286aeb2526e22956f3b0077ef89b97415dda11c6b59a44b3905fc0c1e994',
    'best36_inner_y_b_p_video.csv': '36bc71f28d0c5a29df9b3aff2a774ab97ff4183515f96720f85596d1c3908cbe',
}

def metrics(y, p):
    keep = y != 0
    yt, pt = y[keep] >= 0, p[keep] >= 0
    f1 = 0.
    for cls in (False, True):
        tp = np.sum((yt == cls) & (pt == cls))
        fp = np.sum((yt != cls) & (pt == cls))
        fn = np.sum((yt == cls) & (pt != cls))
        den = 2 * tp + fp + fn
        f1 += np.sum(yt == cls) / len(yt) * (2 * tp / den if den else 0.)
    return dict(Acc7=float(np.mean(np.round(np.clip(y, -3, 3)) == np.round(np.clip(p, -3, 3)))),
                Acc2=float(np.mean(yt == pt)), F1=float(f1),
                MAE=float(np.mean(abs(y - p))), Corr=float(np.corrcoef(y, p)[0, 1]),
                MSE=float(np.mean((y - p) ** 2)))

def load(path):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED[path.name], 'Original CSV SHA mismatch'
    rows = list(csv.DictReader(data.decode('utf-8-sig').splitlines()))
    ids = [r['row_id'] for r in rows]
    assert len(set(ids)) == len(ids)
    y, b, p, f = [np.array([float(r[c]) for r in rows]) for c in ('y', 'b', 'p', 'f')]
    assert all(np.isfinite(x).all() for x in (y, b, p, f))
    return rows, y, b, p, f

def geometry(y, b, p, f):
    eb, ef = y - b, y - f
    d = f - b
    a, bf, c = map(float, (np.mean(eb ** 2), np.mean(ef ** 2), np.mean(eb * ef)))
    den = float(np.mean(d ** 2))
    assert den > 0
    w = float(np.mean(eb * d) / den)
    w_moment = (a - c) / (a + bf - 2 * c)
    assert abs(w - w_moment) < 1e-12
    g = float(np.dot(d, p - b) / np.dot(d, d))
    r_second = c / np.sqrt(a * bf)
    gain = (a - c) ** 2 / den
    oracle = b + w * d
    convex_w = float(np.clip(w, 0, 1))
    excess = den * (g - w) ** 2
    assert abs(a - np.mean((y - oracle) ** 2) - gain) < 1e-12
    assert abs(np.mean((y - p) ** 2) - np.mean((y - oracle) ** 2) - excess) < 1e-6
    return dict(a_MSE_b=a, b_f_MSE_f=bf, c_error_second_crossmoment=c,
                bias_b=float(eb.mean()), bias_f=float(ef.mean()),
                centered_Pearson_error_corr=float(np.corrcoef(eb, ef)[0, 1]),
                uncentered_error_cosine=r_second, second_moment_threshold=float(np.sqrt(a / bf)),
                exact_convex_criterion_c_less_a=bool(c < a), D=den,
                model_gain_from_identity=g, gain_identity_maxerror=float(abs(p - b - g * d).max()),
                diagnostic_only_label_oracle_w=w, diagnostic_only_label_oracle_gain=gain,
                diagnostic_only_label_oracle_MSE=float(np.mean((y - oracle) ** 2)),
                diagnostic_only_convex_w=convex_w,
                diagnostic_only_convex_MSE=float(np.mean((y - b - convex_w * d) ** 2)),
                original_p_MSE=float(np.mean((y - p) ** 2)),
                model_gain_excess_over_label_oracle=excess)

def fit_diagnostics(y, b, f):
    d = f - b
    w = float(np.dot(y - b, d) / np.dot(d, d))
    cal = np.linalg.lstsq(np.column_stack((np.ones(len(b)), b)), y, rcond=None)[0]
    combined = np.linalg.lstsq(np.column_stack((np.ones(len(b)), b, d)), y, rcond=None)[0]
    return w, cal, combined

def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='SAVED_FIT_INNER_NEGATIVE_GAIN_ARITHMETIC_COMPLETE',
                  actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(),
                  fullargv=[sys.executable, *sys.argv], source_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_CSV_SHA=EXPECTED, aggregation='Rows equally weighted; no video resampling or significance claims',
                  official_VAL_TEST_opened=False, model_forward_or_training=False,
                  original_checkpoints_already_FIT_trained_INNER_selected=True,
                  true_OOF_or_new_confirmation=False, models={})
    table = []
    for model in ('best20', 'best36'):
        fit, inner = [load(root / (model + '_' + role + '_y_b_p_video.csv')) for role in ('fit', 'inner')]
        fit_v, inner_v = [{r['video'] for r in x[0]} for x in (fit, inner)]
        assert not fit_v & inner_v
        w, cal, combined = fit_diagnostics(fit[1], fit[2], fit[4])
        rec = dict(FIT_only_parameters=dict(unconstrained_w=w, b_calibration_intercept_slope=cal.tolist(),
                                            combined_intercept_b_delta=combined.tolist()), roles={})
        for role, packet in zip(('fit', 'inner'), (fit, inner)):
            rows, y, b, p, f = packet
            geom = geometry(y, b, p, f)
            preds = dict(original_b=b, original_p=p,
                         FIT_fitted_unconstrained_blend=b + w * (f - b),
                         FIT_fitted_b_calibration=cal[0] + cal[1] * b,
                         FIT_fitted_calibration_plus_delta=combined[0] + combined[1] * b + combined[2] * (f - b))
            rec['roles'][role] = dict(rows=len(rows), videos=len({r['video'] for r in rows}), geometry=geom,
                                     predictions={k: metrics(y, v) for k, v in preds.items()})
            table.append(dict(checkpoint=model, role=role, **geom))
        result['models'][model] = rec
    (out / 'actual_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# 负gain与无约束融合原CSV复核', '',
        '只复算已保存FIT/INNER数组。四CSV先核原SHA；无新训练、模型前向、官方VAL/TEST访问或新独立确认。', '',
        '均按行等权。标签最优w仅为各角色的事后几何诊断，不用于部署或选方法。', '',
        '|模型/角色|误差Pearson r|非中心二阶相关|sqrt(a/b_f)|w*（事后）|模型gain|p MSE|w* MSE|凸混合MSE|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append('|{checkpoint}/{role}|{centered_Pearson_error_corr:.9f}|{uncentered_error_cosine:.9f}|{second_moment_threshold:.9f}|{diagnostic_only_label_oracle_w:.9f}|{model_gain_from_identity:.9f}|{original_p_MSE:.9f}|{diagnostic_only_label_oracle_MSE:.9f}|{diagnostic_only_convex_MSE:.9f}|'.format(**row))
    lines += ['', 'r若指Pearson而a、b_f指MSE，存在非零偏差时不能直接套r<sqrt(a/b_f)；精确判据是原二阶矩c<a，或使用非中心相关c/sqrt(a*b_f)。本次四行c均大于a，凸混合最优均为b。', '',
              '以下权重只在FIT拟合后原样用于INNER；原任务模型已见过FIT且用INNER选checkpoint，因此仍是描述性对照，不是真实OOF泛化证据。', '',
              '|模型/INNER固定方法|Acc7|Acc2|F1|MAE|Corr|MSE|', '|---|---:|---:|---:|---:|---:|---:|']
    for model, rec in result['models'].items():
        for name, m in rec['roles']['inner']['predictions'].items():
            lines.append('|'+model+'/'+name+'|'+'|'.join(f'{m[k]:.9f}' for k in ('Acc7','Acc2','F1','MAE','Corr','MSE'))+'|')
    lines += ['', '高误差相关与负w是校准/反向外插的提示，不独立证明供体完全无信息或无样本特异变化。b为原多模态池化读出，不是文本专家，也不是真正同流关闭消息p0。', '',
              'best36 INNER约0.03344 MSE的差是该集合事后最优gain与当前gain的几何差，不等于一个可泛化校准层已经挽回同等损失。', '',
              '实际运行来源、原CSV SHA、误差均值、二阶矩、参数、五项完整精度及限制见actual_result.json。']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], table=table,
                         inner={k:v['roles']['inner']['predictions'] for k,v in result['models'].items()}), ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.out)

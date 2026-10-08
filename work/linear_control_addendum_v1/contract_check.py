"""Synthetic baseline algebra and source/role guards, local NumPy only."""
import datetime, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from linear_controls import LinearControls, error_geometry
from oof_selector import PREDECESSORS, weights

def rejected(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Invalid packet accepted')

def main():
    rng = np.random.default_rng(128)
    ids = [f'fit_{v}[{j}]' for v in range(10) for j in range(3)]
    cal = [f'cal_{v}[0]' for v in range(2)]
    allowed = ids+cal
    lineage=[]
    for k in range(5):
        held=[s for s in ids if s.split('[')[0] in {f'fit_{2*k}',f'fit_{2*k+1}'}]
        train=[s for s in ids if s not in held]
        lineage.append(dict(held_ids=held,predecessors={n:dict(train_ids=train,state_SHA='a'*64) for n in PREDECESSORS}))
    p0=rng.normal(size=len(ids));d=rng.normal(size=len(ids));p1=p0+d
    y=.2+1.1*p0-.4*d
    model=LinearControls().fit(y,p0,p1,ids,allowed,cal,lineage)
    exact=float(abs(model.predict(p0,p1)['OOF_calibration_plus_delta']-y).max())
    assert exact<1e-12
    yn=p0-.6*d
    blend=LinearControls().fit(yn,p0,p1,ids,allowed,cal,lineage)
    assert abs(blend.blend_w+.6)<1e-12
    old=blend.predict(p0,p1)
    error=float(abs(old['OOF_unconstrained_blend']-yn).max())
    assert error<1e-12
    assert all(np.array_equal(old[k],blend.predict(p0,p1)[k]) for k in old)
    # Pure scaling: extra-delta arm has no new direction beyond affine p0.
    scaling=LinearControls().fit(y,p0,.5*p0,ids,allowed,cal,lineage)
    assert scaling.delta_gamma == 0.
    fallback=LinearControls().fit(y,p0,p0,ids,allowed,cal,lineage)
    assert fallback.blend_w == 0. and fallback.delta_gamma == 0.
    assert np.array_equal(fallback.predict(p0,p0)['OOF_unconstrained_blend'],p0)
    rejected(lambda:LinearControls().predict(p0,p1))
    bad=json.loads(json.dumps(lineage));bad[0]['predecessors']['encoder']['train_ids'].append(bad[0]['held_ids'][0])
    rejected(lambda:LinearControls().fit(y,p0,p1,ids,allowed,cal,bad))
    rejected(lambda:LinearControls().fit(y,p0,p1,ids,allowed,ids[:1],lineage))
    geom=error_geometry(yn,p0,p1,weights(ids))
    assert abs(geom['diagnostic_only_label_optimal_w']+.6)<1e-12
    weighted=weights(ids)
    independent=float(weighted@((yn-p0)*(p1-p0))/(weighted@((p1-p0)**2)))
    assert abs(geom['diagnostic_only_label_optimal_w']-independent)<1e-12
    result=dict(status='LOCAL_SYNTHETIC_LINEAR_CONTROLS_CONTRACT_COMPLETE',
                actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
                fullargv=[sys.executable,*sys.argv],synthetic_only=True,
                source_SHA={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ('contract_check.py','linear_controls.py','oof_selector.py')},
                affine_plus_delta_exact_error=exact,negative_blend_exact_error=error,
                zero_delta_exact_OFF=True,affine_delta_extra_arm_exact_calibration=True,
                FIT_video_and_CAL_leakage_rejected=True,evaluation_predictions_take_no_labels=True,
                real_OOF_train_complete=False,new_VAL_TEST_scores=False,original_server_CPU=False)
    Path(sys.argv[1]).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()

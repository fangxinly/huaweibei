import argparse,ast,datetime,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path('work/innovation_v1').resolve()))
from contract import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args();stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
c=read('work/innovation_v1_current.json');D=Path(c['D_actual']);audit=read(D/'B_train_audit/actual_D_verification.json');assert audit['natural_exit']==0
root=Path('work')/('innovation_v1_score_'+stamp);root.mkdir();b=root/'source';b.mkdir()
for src in (Path('work/score_innovation_pilot.py'),Path(c['bundle'])/'sentiment_metrics_careflow_v1.py'):
    ast.parse(src.read_text(encoding='utf8'));shutil.copy2(src,b/src.name)
labels=Path('outputs/best20_best36_FIT_INNER_review_20261008T012035Z/best36_inner_y_b_p_video.csv')
assert sha(labels)=='36bc71f28d0c5a29df9b3aff2a774ab97ff4183515f96720f85596d1c3908cbe'
t=D/'A_train/extracted/out';plan=dict(status='FIXED_ALL_FIVE_INNER_ONCE_SCORE_FROZEN',actualclock_UTC=a.clock,D=str(D),methods=read(Path(c['bundle'])/'execution_protocol.json')['methods'],source_SHA={x.name:sha(x) for x in b.iterdir()},train_D_verification_SHA=sha(D/'A_train/actual_D_verification.json'),CPU_D_verification_SHA=sha(D/'B_train_audit/actual_D_verification.json'),prediction_SHA=sha(t/'frozen_INNER_predictions.npz'),gate_details_SHA=sha(t/'frozen_INNER_gate_details.npz'),labels_CSV=str(labels.resolve()),labels_CSV_SHA=sha(labels),bootstrap_seed=128,bootstrap_repeats=5000,weak_definition='abs(y)<=1',no_fit_no_selection=True,not_official_TEST_or_formal_CaReFlow=True)
write(b/'scoring_protocol.json',plan);target=D/('score_'+stamp);target.mkdir();shutil.copytree(b,target/'source');pointer=dict(root=str(root.resolve()),source=str(b.resolve()),D=str(target),plan_SHA=sha(b/'scoring_protocol.json'));write('work/innovation_v1_score_current.json',pointer);print(pointer)

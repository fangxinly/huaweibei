import sys,json,shutil,datetime
from pathlib import Path
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root/'work/anchored_increment_candidate'))
from common import sha,read,write
ref=read(root/'work/anchored_increment_pointer.json');s=Path(ref['source']);d=Path(ref['D']);mode=sys.argv[1];clock=sys.argv[2];p=read(s/'increment_protocol.json')
def qualified(name):
    v=read(d/name/'actual_D_verification.json');assert v['natural_exit']==0;return dict(path=str(d/name),SHA=sha(d/name/'actual_D_verification.json'),archive_SHA=v['archive_SHA'])
if mode=='train':
    n={k:qualified(k) for k in ('A_native','B_native')};cache=qualified('A_cache');cr=read(d/'A_cache/extracted/out/cache_result.json');assert cr['status']=='ORIGINAL_BEST36_FIRST_EULER_CACHE_COMPLETE';assert cr['selected_state_SHA']==p['parent']['selected_state_SHA'];assert cr['parameters_buffers_rng_unchanged'];assert max(cr['p0_saved_original_maxerror'].values())<=p['cache_p0_tolerance']
    p.update(real_train_enabled=True,allowed_stages=['train'],native_D_other_CPU_reference=n,cache_D_reference=cache,cache_reference={'SHA':cr['cache_SHA'],'bytes':cr['cache_bytes'],'D_path':str(d/'A_cache/extracted/out/original_first_states.npz')},tail_reference={'SHA':cr['tail_SHA'],'D_path':str(d/'A_cache/extracted/out/original_tail.pt')},single_training_launch_token='/data/coding/anchored_increment_source_20261008T024739Z/train_once.lock',actualclock_execution_freeze_UTC=clock)
    filename='increment_train_protocol.json'
elif mode=='audit':
    train=qualified('A_train');tr=read(d/'A_train/extracted/out/train_result.json');assert tr['status']=='ORIGINAL_BEST36_MESSAGE_INCREMENT_FIXED20_COMPLETE';assert tr['updates']==940 and tr['inner_labels_not_indexed']
    tp=s/'increment_train_protocol.json';p=read(tp);p.update(allowed_stages=['audit'],training_plan_SHA=sha(tp),actualclock_audit_freeze_UTC=clock,training_D_reference=train,changed_state_reference={'SHA':tr['changed_state_SHA'],'D_path':str(d/'A_train/extracted/out/complete_changed_training_state.pt')},prediction_reference={'SHA':tr['prediction_SHA'],'D_path':str(d/'A_train/extracted/out/fixed_final_predictions.npz')});filename='increment_audit_protocol.json'
else:raise ValueError(mode)
assert shutil.disk_usage(d).free>=1000000000;assert not (s/filename).exists();write(s/filename,p);shutil.copy2(s/filename,d/filename);h=sha(s/filename);write(d/(mode+'_plan_reference.json'),dict(plan=str(s/filename),SHA=h,actualclock=clock));print(json.dumps(dict(plan=str(s/filename),SHA=h,cache=p['cache_reference'],tail=p['tail_reference'])))

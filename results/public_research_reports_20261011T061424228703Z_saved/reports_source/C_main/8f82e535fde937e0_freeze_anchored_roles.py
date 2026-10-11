import sys,json,shutil,zipfile,hashlib,ast,datetime
from pathlib import Path
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root/'work/anchored_increment_candidate'))
from common import sha,read,write
stamp=sys.argv[1];clock=sys.argv[2];old=root/'work/anchored_increment_source_20261008T024739Z';p=read(old/'increment_audit_protocol.json');parentD=Path(p['parent']['D_path']);assert sha(parentD)==p['parent']['full_SHA'];assert sha(p['changed_state_reference']['D_path'])==p['changed_state_reference']['SHA']
s=root/'work'/('anchored_VAL_TEST_eval_'+stamp);s.mkdir();d=Path('D:/CodexBackups/selective_flow_20261003_1105')/('anchored_VAL_TEST_actual_'+stamp);d.mkdir();assert shutil.disk_usage(d).free>=1000000000
for n,h in p['source_sha256'].items():assert sha(old/n)==h;shutil.copy2(old/n,s/n)
shutil.copy2(root/'work/anchored_increment_inference_20261008T025648Z/inference_upgrade.py',s/'inference_upgrade.py')
for f in (root/'work/anchored_role_eval_candidate').glob('*.py'):shutil.copy2(f,s/f.name)
for f in s.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
split=read(s/'split.json');canonical=split['canonical_row_ids'];assert len(canonical)==2195;ids={'VAL':canonical[1281:1510],'TEST':canonical[1510:]};assert len(ids['VAL'])==229 and len(ids['TEST'])==685;roles={r: set(split['folds'][0]['row_ids'][r]) for r in ('fit','inner','outer')};overlap={name:{r:sum(s in group for s in ss) for r,group in roles.items()} for name,ss in ids.items()};rolemap={s:r for r,group in roles.items() for s in group}
oldD=Path('D:/CodexBackups/selective_flow_20261003_1105/anchored_increment_actual_20261008T024739Z');joint=read(oldD/'final_delivery.json');assert joint['all_four_connections_closed_natural0'];assert sha(oldD/'A_train/actual_D_verification.json')==p['training_D_reference']['SHA']
p.update(status='FIXED_ORIGINAL_INCREMENT_OFFICIAL_VAL_TEST_INFERENCE_SCORING_PROTOCOL_FROZEN',actualclock_scope_freeze_UTC=clock,allowed_stages=['infer'],real_train_enabled=False,source_sha256={f.name:sha(f) for f in s.iterdir()},official_role_IDs=ids,official_role_training_overlap=overlap,merged_ID_to_role=rolemap,adapter_state_SHA='b9906e1b9c7419c33a79dbba1d34d6d0e69f82dc02a9f7fa3764d3e2c271ae77',tail_state_SHA='2a0ffcaea5f424fd6e58cb54d9bd569d1be639b164cbb81ef2d80ec1cc7bdebe',adapter_remote_path='/data/coding/anchored_increment_train_A_20261008T024739Z/out/complete_changed_training_state.pt',direct_forward_tolerance=2e-6,original_model_D_CPU_qualification={'delivery':str(oldD/'final_delivery.json'),'SHA':sha(oldD/'final_delivery.json'),'training_A_D_verification':p['training_D_reference'],'CPU_B_archive_SHA':read(oldD/'B_audit/actual_D_verification.json')['archive_SHA']},original_D_CPU_score_gate=None,infer_once_token='/data/coding/anchored_VAL_TEST_eval_'+stamp+'/infer_once.lock',score_once_token='/data/coding/anchored_VAL_TEST_eval_'+stamp+'/score_once.lock',stage_budget_seconds={'infer':900,'audit':600,'score':180},evaluation_only=True,all_five_same_fixed_checkpoint=True,metrics_rule='Author clip-round Acc7 all rows; nonzero truth/pred>=0 Acc2 and supportweightedF1; raw MAE/Pearson. No calibration, training, hyperparameter or checkpoint change',official_TEST_not_independent_because_original_merged_training=True,user_scope='User explicitly requires VAL and TEST all five; score current fixed model even though overlap exists; disclose counts')
tr=read(oldD/'A_train/extracted/out/train_result.json');p['adapter_state_SHA']=tr['adapter_state_SHA'];write(s/'official_roles_protocol.json',p)
zp=d/'frozen_VAL_TEST_eval_source.zip'
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
    for f in s.iterdir():z.write(f,f.name)
with zipfile.ZipFile(zp) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for f in s.iterdir():assert hashlib.sha256(z.read(f.name)).hexdigest()==sha(f)
r=dict(source=str(s),D=str(d),ZIP=str(zp),ZIP_SHA=sha(zp),plan_SHA=sha(s/'official_roles_protocol.json'),actualclock=clock,official_role_training_overlap=overlap,D_free_bytes=shutil.disk_usage(d).free,CRC_SHA_unique=True);write(d/'source_reference.json',r);write(root/'work/anchored_roles_pointer.json',r);print(json.dumps(r))

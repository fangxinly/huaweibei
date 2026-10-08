"""Copy a reviewed official pipeline; change only candidate installation/export."""
import ast,hashlib,json,shutil
from pathlib import Path
base=Path(__file__).resolve().parents[1]
original=base/'work/official_anchored_upgrade_20261008T053146Z'
out=base/'work/polarity_intensity_official_pipeline_preparation'
out.mkdir(exist_ok=False)
for p in original.iterdir():
    if p.suffix in ('.py','.npy'):shutil.copyfile(p,out/p.name)
for p in (base/'work/polarity_intensity_candidate_v1').glob('*.py'):shutil.copyfile(p,out/p.name)
def change(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
text=(original/'train_official.py').read_text()
text=change(text,'from official_upgrade import OfficialUpgrade,objective','from candidate_adapter import make_candidate,optimizer_groups,CandidateTail,candidate_objective as objective')
text=change(text,'from incremental_message import OriginalTail','')
text=change(text,"assert p['native_CPU_D_qualification'];assert p['epochs']", "assert p['native_CPU_D_qualification'] and p['candidate_native_D_B_qualification'] and p['candidate_full_encoder_precheck_D_B_qualification'];assert p['candidate_mode'] in ('regression_aux','factorized_aux');assert p['epochs']")
text=change(text,'model.dberta.own_flow=OfficialUpgrade()','model.dberta.own_flow=make_candidate(p[\'candidate_mode\'])')
start=text.index('    gain=model.dberta.own_flow.gain;special=')
end=text.index('    opt=torch.optim.AdamW(groups)',start)
text=text[:start]+'    groups,named=optimizer_groups(model)\n'+text[end:]
text=text.replace("p['source_sha256']['official_upgrade.py']","p['source_sha256']['polarity_intensity_flow.py']")
text=change(text,"tail=OriginalTail(model.dberta);torch.save({'state':tail.state_dict()},a.out/'selected_original_tail.pt');torch.save({'state':cpu(model.dberta.own_flow.message.state_dict())},a.out/'selected_message.pt')", "tail=CandidateTail(model.dberta);torch.save({'state':tail.state_dict(),'mode':p['candidate_mode']},a.out/'selected_original_tail.pt');torch.save({'state':cpu(model.dberta.own_flow.core.message.state_dict())},a.out/'selected_message.pt')")
text=text.replace('CLEAN_OFFICIAL_ANCHORED_UPGRADE_TRAIN100_COMPLETE','CLEAN_OFFICIAL_POLARITY_INTENSITY_TRAIN100_COMPLETE')
text=change(text,"metadata=dict(plan_SHA=a.plan_sha", "metadata=dict(candidate_mode=p['candidate_mode'],fixed_loss='Huber1(p,y)+.1Huber1(m,abs(y))+.1weighted_signBCE+.01context2',plan_SHA=a.plan_sha")
ast.parse(text);(out/'train_official.py').write_text(text,encoding='utf-8')
text=(original/'infer_official.py').read_text()
text=change(text,'from official_upgrade import OfficialUpgrade','from candidate_adapter import make_candidate,CandidateTail')
text=change(text,'from incremental_message import OriginalTail','')
text=change(text,"assert p['training_D_B_CPU_gate'];a.out.mkdir()", "assert p['training_D_B_CPU_gate'] and p['candidate_training_D_B_CPU_gate'];assert p['candidate_mode'] in ('regression_aux','factorized_aux');a.out.mkdir()")
text=change(text,"full=torch.load(cp,map_location='cpu');selected=full['selected_model']", "full=torch.load(cp,map_location='cpu');assert full['metadata']['candidate_mode']==p['candidate_mode'];selected=full['selected_model']")
text=change(text,'model.dberta.own_flow=OfficialUpgrade()','model.dberta.own_flow=make_candidate(p[\'candidate_mode\'])')
text=change(text,'tail=OriginalTail(model.dberta)','tail=CandidateTail(model.dberta)')
start=text.index('zero=src.new_zeros(')
end=text.index('\n                err=',start)
text=text[:start]+'explicit,p0=tail(src,mask)'+text[end:]
text=text.replace('TRAIN_ONLY_SELECTED_UPGRADE_VAL_TEST_PREDICTION_COMPLETE','TRAIN_ONLY_SELECTED_POLARITY_INTENSITY_VAL_TEST_PREDICTION_COMPLETE')
ast.parse(text);(out/'infer_official.py').write_text(text,encoding='utf-8')
for p in out.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
receipt=dict(status='OFFICIAL_PIPELINE_SOURCE_PREPARATION_ONLY_NOT_EXECUTED',
    original_root=str(original),new_root=str(out),original_files_unchanged=True,
    candidate_full_encoder_precheck_D_B_qualification=False,
    real_TRAIN_VAL_TEST=False,shared_budget='seed128/TRAIN1281/100epochs/4000updates/batch32/drop_lastTrue/shared orders',
    selection='VAL229 batch128/101 mean MSE strict earliest',TEST_not_used_for_selection=True,
    per_file_SHA256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
(out/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],files=len(receipt['per_file_SHA256'])),ensure_ascii=False))

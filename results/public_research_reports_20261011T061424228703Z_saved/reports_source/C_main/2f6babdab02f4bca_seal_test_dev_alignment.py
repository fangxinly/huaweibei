import json,hashlib,zipfile,shutil,sys
from pathlib import Path
root=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
out=root/'outputs'
stamp=sys.argv[1] if len(sys.argv)>1 else '20261007T104048Z'
clock=sys.argv[2] if len(sys.argv)>2 else '2026-10-07 10:40:48 UTC'
base=Path('D:/CodexBackups/selective_flow_20261003_1105')/('test_dev_history_alignment_'+stamp)
base.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['CaReFlow五指标统一评价与实际DEV结果.json','完整流匹配U_W头201一次开发五指标实际结果.json',
 '正式双方fullTRAIN100固定best一次DEV五项实际结果.json','正式双方TEST685一次五项实际结果.json',
 '固定双方TRAIN_DEV退化诊断.json','有限任务风险C2修订实验分析.md',
 '有限任务风险C2修订100轮冻结计划.json','保留旧0.598信号的下一项受控训练方向.md']
src={n:dict(path=str(out/n),sha256=sha(out/n)) for n in names}
old=json.loads((out/names[0]).read_text(encoding='utf-8'))['result']
dev=json.loads((out/names[2]).read_text(encoding='utf-8'))['methods']
test=json.loads((out/names[3]).read_text(encoding='utf-8'))['metrics']
data={'status':'SAVED_AGGREGATE_ALIGNMENT_NO_NEW_TEST_SCORE_OR_FORWARD',
 'actualclock_record_utc':clock,'sources':src,
 'old_DEV229':old['rows'],'old_provenance':old['provenance'],
 'matched_DEV229':{k:v['fixed_DEV_five'] for k,v in dev.items()},'same_selected_TEST685':test,
 'same_model_DEV_to_TEST_MAE':{k:test[k]['MAE']-v['fixed_DEV_five']['MAE'] for k,v in dev.items()},
 'same_DEV229_oldA_to_newF_MAE_change':dev['minimal_fixed_F']['fixed_DEV_five']['MAE']-old['rows']['Own_A_saved_default_DEV']['MAE'],
 'oldA_is_same_checkpoint_as_newF':False,'old_seed':91817,'new_seed':128,
 'old_train_scope':'Frozen backbone/flow/reader/decoder; train520506 donor plus matched head',
 'old_teacher_fullTRAIN_fit_DEV_selected':True,
 'old_teacher_legal_initialization_for_merged_fivefold':False,
 'new_train_scope':'Full end-to-end minimal_fixed_F,364 Adam parameter states',
 'confirmed_causal_reason':False,'old0_59_positive_DEV_signal_retained':True,
 'TEST_used_to_select_structure_or_coefficients':False,'new_TEST_label_reads':False,
 'current_pilot_frozen_source_unchanged':True,'full_fivefold_complete':False}
target=out/'TEST与既有0.59结果对齐分析.json'
target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
payload=base/'payload';payload.mkdir()
for n in names+['TEST与既有0.59结果对齐分析.md','TEST与既有0.59结果对齐分析.json']:
 shutil.copy2(out/n,payload/n)
shutil.copy2(Path(__file__),payload/Path(__file__).name)
manifest={p.name:sha(p) for p in payload.iterdir()}
(payload/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=base/'complete_local_alignment.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(payload.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
receipt={'status':'LOCAL_ALIGNMENT_COMPLETE_SHA_CRC_UNIQUE','not_remote_capture':True,
 'zip_sha256':sha(archive),'report_sha256':sha(target),'member_count':len(manifest)+1}
(base/'actual_local_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))

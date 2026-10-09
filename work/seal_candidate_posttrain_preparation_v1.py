"""Exact CPU-native originals and isolated C runtime proof, small C/D save."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
def read(f):return json.loads(f.read_text(encoding='utf8'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
source=base/'work/candidate_posttrain_preparation_20261009T004259Z';plan=read(source/'native_plan.json');files=[source/'source.zip',source/'native_plan.json',base/'work/freeze_candidate_posttrain_stage_v1.py',base/'work/prepare_candidate_pureCPU_runtime_v1.py'];nodes={}
for node in ('A','B'):
    cap=read(ev/(node+'_candidate_posttrain_native_capture.json'));exit=read(ev/(node+'_candidate_posttrain_native_exit.json'));result=read(ev/(node+'_candidate_posttrain_native_result.json'));archive=ev/(node+'_candidate_posttrain_native_original.zip')
    assert cap['natural_exit']==exit['natural_exit']==0 and sha(archive)==cap['archive_SHA']
    assert result['status']=='CANDIDATE_BOTH_MODES_SAVED_TAIL_AND_INDEPENDENT_FIVE_CPU_PASSED' and not result['real_data_or_labels_indexed']
    assert result['source_SHA']==sha(source/'qualify_candidate_posttrain_v1.py')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        members=json.loads(z.read('member_SHA.json'))
        for name,digest in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
        for f in source.glob('*.py'):assert sha(f)==members['original_source/'+f.name]
    nodes[node]=dict(capture=cap,exit=exit,result=result)
    for suffix in ('capture.json','exit.json','result.json','original.zip'):files.append(ev/(node+'_candidate_posttrain_native_'+suffix))
    progress_path=ev/(node+'_recovery400_posttrainprep_progress.json');progress=read(progress_path);old=read(ev/(node+'_interrupted_history_original.json'));r=old[progress['epoch']-1]
    for name in ('state_SHA','prediction_SHA','DEV_batch_MSE'):assert progress[name]==r[name]
    nodes[node]['actual_training_progress']=progress;files.append(progress_path)
cap=read(ev/'C_candidate_pureCPU_runtime_capture.json');result=read(ev/'C_candidate_pureCPU_runtime_result.json');exit=read(ev/'C_candidate_pureCPU_runtime_exit.json');archive=ev/'C_candidate_pureCPU_runtime_original.zip'
assert cap['natural_exit']==exit['natural_exit']==0 and sha(archive)==cap['archive_SHA']
assert result['runtime_versions']==plan['runtime_versions'] and result['CPU_only'] and not result['model_data_download']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,digest in json.loads(z.read('member_SHA.json')).items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
for suffix in ('capture.json','exit.json','result.json','original.zip'):files.append(ev/('C_candidate_pureCPU_runtime_'+suffix))
files.append(ev/'C_pureCPU_readonly_preflight.json')
assert shutil.disk_usage('C:/').free>200_000_000 and shutil.disk_usage('D:/').free>500_000_000
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('candidate_posttrain_native_'+a.tag);dest.mkdir();archive=dest/'all_small_native_runtime_and_prepared_source_originals.zip'
members={f.name:sha(f) for f in files};assert len(members)==len(files)
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.name)
    z.writestr('member_SHA.json',json.dumps(members))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,digest in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
proof=dict(status='TWO_NODE_CANDIDATE_POSTTRAIN_NATIVE_AND_C_ISOLATED_RUNTIME_C_D_PASSED',actualclock_UTC=a.stamp,nodes=nodes,C_runtime=result,C_runtime_capture=cap,prepared_source=str(source),prepared_source_zip_SHA=sha(source/'source.zip'),prepared_source_plan_SHA=sha(source/'native_plan.json'),D_archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,actual_final100_stage_not_yet_frozen=True,no_new_VAL_TEST_five=True)
(ev/'candidate_posttrain_preparation_C_D_verification.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
state_path=base/'outputs/自主优化实际接续.json';state=read(state_path);pair=state['polarity_intensity_pair_v2'];pair['status']='MATCHED_PAIR_DETACHED_RECOVERY400_RUNNING';pair['candidate_posttrain_preparation']=proof
for node in ('A','B'):pair['recovery400_actual'][node]['latest_progress_original']=nodes[node]['actual_training_progress']
state['updated_at_utc']=a.stamp;state['actualclock_before_snapshot_UTC']=a.stamp
state['sessions_current']['C_SSH']=58876;state['sessions_current']['C_SFTP']=54162
state['current_capacity']={n:shutil.disk_usage(n+':/').free for n in ('C','D')}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
short=a.stamp+' A/B两臂最新各35轮1400步，原参数/预测SHA/VAL选模MSE与中断前轨迹完全一致；第800步完整可变状态已落盘，训练健康继续。完成后CandidateTail核验与VAL/TEST五项源已适配，两节点纯合成CPU自然0，不是新数据结果或真实4000审核。C真实UUID/空compute/47GB空间核过，独立新CPU依赖环境24项含Torch2.1cu121原版本完全一致，setup自然0/C-D完整小原件SHA CRC唯一过，无模型或数据下载、不改旧环境。100完整原件先Release，CPU用C纯CPU环境逐成员线性SHA/CRC/唯一核验并仅临时解出finalPT，避免ZIP seek；实际posttrain计划仍须真实100自然0及完整保存后冻结。尚无新最终VAL/TEST五项，整体未完成。\n\n'
status=base/'outputs/研究接续状态.md';status.write_text(short+status.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(status=proof['status'],D_archive=str(archive),archive_SHA=proof['archive_SHA'],bytes=proof['archive_bytes'],no_new_final_five=True)))

"""Preserve the explicit C identity guard and latest original running receipts."""
import argparse,ast,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
from prepare_github_source_upload import sanitize
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z';repo=base/'github_upload_20261008T133346Z'
git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
def read(f):return json.loads(f.read_text(encoding='utf8'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def command(args,data=None):return subprocess.check_output([git]+args,cwd=repo,input=data)
assert not command(['status','--porcelain']).strip()
helper=base/'work/freeze_candidate_posttrain_stage_v1.py';ast.parse(helper.read_text(encoding='utf8'));files=[helper];nodes={}
for node in ('A','B'):
    progress_path=ev/(node+'_recovery400_finalprep_current_progress.json');checkpoint_path=ev/(node+'_recovery400_finalprep_current_checkpoint.json');progress=read(progress_path);checkpoint=read(checkpoint_path);old=read(ev/(node+'_interrupted_history_original.json'))
    assert progress['steps']==progress['epoch']*40 and checkpoint['steps']==1600
    for name in ('state_SHA','prediction_SHA','DEV_batch_MSE'):assert progress[name]==old[progress['epoch']-1][name]
    nodes[node]=dict(progress=progress,checkpoint_ready=checkpoint,recorded_trajectory_exact=True,large_periodic_checkpoint_not_peer_CPU_audited=True);files.extend((progress_path,checkpoint_path))
assert shutil.disk_usage('C:/').free>200_000_000 and shutil.disk_usage('D:/').free>500_000_000
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('candidate_posttrain_C_identity_guard_'+a.tag);dest.mkdir();archive=dest/'current_helper_and_running_original_receipts.zip';members={f.name:sha(f) for f in files}
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.name)
    z.writestr('member_SHA.json',json.dumps(members))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,digest in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
proof=dict(actualclock_UTC=a.stamp,nodes=nodes,freeze_helper_SHA=sha(helper),C_identity_bound_to_actual_preserved_runtime=True,actual_final4000_plan_not_yet_frozen=True,D_archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,no_new_final_five=True)
proof_path=ev/'candidate_posttrain_C_identity_and_running_C_D_verification.json';proof_path.write_text(json.dumps(proof,indent=2),encoding='utf8')
files.extend((proof_path,Path(__file__)))
manifest=read(repo/'source_manifest.json');records={r['path']:r for r in manifest['records']};changed=[]
for f in files:
    relative=('work/'+f.name) if f.suffix=='.py' else ('results/polarity_intensity_continuation/'+f.name)
    raw=f.read_bytes();clean,changes=sanitize(raw);target=repo/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean)
    if f.suffix=='.py':ast.parse(clean.decode('utf-8-sig'))
    records[relative]=dict(path=relative,origin=f.relative_to(base).as_posix(),kind='source' if f.suffix=='.py' else 'result',bytes=len(clean),original_sha256=hashlib.sha256(raw).hexdigest(),published_sha256=hashlib.sha256(clean).hexdigest(),sanitizations=changes);changed.append(relative)
manifest['records']=[records[k] for k in sorted(records)];(repo/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');changed.append('source_manifest.json')
for relative in changed:
    data=(repo/relative).read_bytes();oid=command(['hash-object','-w','--stdin'],data).decode().strip();command(['update-index','--add','--cacheinfo','100644',oid,relative]);assert command(['show',':'+relative])==data
command(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit','-m','Bind final CPU stage to verified C identity and retain running receipts'])
command(['push','origin','main']);head=command(['rev-parse','HEAD']).decode().strip();assert head==command(['ls-remote','origin','refs/heads/main']).decode().split()[0] and not command(['status','--porcelain']).strip()
git_receipt=dict(status='SOURCE_RESULTS_EXACT_BLOB_PUSH_VERIFIED',actualclock_before_publication_UTC=a.stamp,commit=head,files=len(changed),no_new_final_five=True)
(ev/'candidate_posttrain_identity_GitHub_receipt.json').write_text(json.dumps(git_receipt,indent=2),encoding='utf8')
state_path=base/'outputs/自主优化实际接续.json';state=read(state_path);pair=state['polarity_intensity_pair_v2'];pair['posttrain_C_identity_guard']=proof
for node,row in nodes.items():
    pair['recovery400_actual'][node]['latest_progress_original']=row['progress'];pair['recovery400_actual'][node]['latest_mutable_checkpoint_ready_original']=row['checkpoint_ready']
state['github_source']=git_receipt;state['updated_at_utc']=a.stamp;state['actualclock_before_snapshot_UTC']=a.stamp;state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
short=a.stamp+' 两臂最新各41轮/1640步，1600步完整可变检查点原ready已取回；原参数/预测SHA及VAL选模MSE与旧中断轨迹一致。完成后CandidateTail与五项源两节点合成CPU自然0/C-D原件过；C新独立纯CPU24项版本相同/约46.7GB空间已核过，无模型数据下载。stage冻结helper将C身份明确绑定已存原runtime UUID，真实100及其Release完成后才冻结执行，不冒4000 CPU或新最终五项。源码/报告实际GitHub main '+head[:8]+'。整体未完成。\n\n'
status=base/'outputs/研究接续状态.md';status.write_text(short+status.read_text(encoding='utf8'),encoding='utf8')
next_text='# 当前接续\n\n'+short+'\n训练健康继续：A3079/B3696 detached，root A/B_candidate_recovery400_actual_20261009T002805Z；不重启、不重跑前400/16。取最新progress、natural_exit、capture原件，完成立即Release完整原ZIP。\n\n候选完成后源码：work/candidate_posttrain_preparation_20261009T004259Z，A/B原CPU合成资格见自主优化JSON candidate_posttrain_preparation。使用work/freeze_candidate_posttrain_stage_v1.py，实际候选final100单独证据目录需保存A100/B100_capture_receipt/natural_exit/publication_receipt/publication_client_exit/training_result/member_SHA.json；不能借已完成MSE的A100旧证据。--stage audit --training-node A或B --execution-node C --tools上述候选源码 --evidence新候选证据目录 --out新实际冻结目录 --stamp actualclock。完成后纯CPU实际执行仍要新fresh UUID/source/runtime/空间/时余核验。\n\nC纯CPU assets=/data/coding/C_candidate_pureCPU_runtime_actual_20261009T004259Z，.venv/bin/python；无模型数据，24项依赖与A/B一致。原完整ZIP一次公开下载，全成员线性SHA/CRC/唯一性，仅临时解出finalPT，原CPU4000完成后只移除明确新临时副本，完整ZIP保留。再--stage infer/score固定同checkpoint官方VAL229/TEST685预测先Release及原数组审核、五项一次及另一节点原NPZ独立核算。C已具备准备但不是新4000审核完成。\n\nSSH A42487/B4798/C58876，SFTP A72818/B55421/C54162当前idle；原训练detached，全部旧closed/Unknown ID禁复用。旧慢ZIP CPU2855/3410仍未确认完成，不强杀。原10min ACTIVE、保守租期UTC15:00、至少2h保存门、C/D小件/大原件Release、禁删旧冻源保持。不据TEST改结构，已完成MSE100/201/FIT232/reference100/232禁重跑。\n'
(base/'NEXT_EXECUTION.md').write_text(next_text,encoding='utf8')
print(json.dumps(dict(github=git_receipt,D_archive=str(archive),archive_SHA=proof['archive_SHA'],no_new_final_five=True)))

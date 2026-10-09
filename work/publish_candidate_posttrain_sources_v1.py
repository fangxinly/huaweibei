"""Publish native-qualified posttrain source; never imply final candidate scores."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
from prepare_github_source_upload import sanitize
base=Path(__file__).resolve().parents[1];repo=base/'github_upload_20261008T133346Z';ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);a=p.parse_args()
def command(args,data=None):return subprocess.check_output([git]+args,cwd=repo,input=data)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
proof=json.loads((ev/'candidate_posttrain_preparation_C_D_verification.json').read_text(encoding='utf8'))
assert proof['status']=='TWO_NODE_CANDIDATE_POSTTRAIN_NATIVE_AND_C_ISOLATED_RUNTIME_C_D_PASSED'
source=Path(proof['prepared_source']);assert sha(source/'source.zip')==proof['prepared_source_zip_SHA']
report=base/'outputs/极性强度完成后评价准备实际结果.md'
report.write_text('# 极性／强度两臂：完成后评价流程核验\n\n记录时钟：'+a.stamp+'。\n\n两臂原流训练仍在执行，尚无新最终VAL／TEST五项。旧100过程曾中断；恢复后的参数和预测SHA与原轨迹一致，额外重放计算成本保留。\n\n旧完成后CPU脚本只支持OriginalTail，与新增极性／强度头不兼容。现在保存后CandidateTail结构、完整模型与尾部参数联结、345个Adam的4000步、100轮官方VAL128／101批MSE选模、相同选中checkpoint的VAL和TEST五项流程已接入新源码。真实完整4000审核尚未执行；实际stage计划必须等100自然退出、完整原件Release保存后才冻结。\n\nA／B两节点的纯合成CPU检查均自然0：两种模式保存后读出完全重现，消息开／关非零，dummy标签0与7预测一致，参数与RNG不变；作者五项与独立sklearn／scipy算术一致。该核验没有读取真实数据或标签，也不是性能结果。\n\nC新建独立CPU环境，固定依赖版本与A／B一致，Torch2.1.0+cu121；实测剩余约46.7GB。没有下载模型或数据，没有修改旧环境。大原件审计将线性读取ZIP全部成员核SHA／CRC／唯一性，只临时解出完整final状态直接torch.load；成功后移除明确任务临时解出副本，完整原ZIP永久保留。\n\n新推理检查还会核对官方VAL重新前向与训练时冻结预测，之后固定同一checkpoint计算VAL229／TEST685的Acc7、Acc2、F1、MAE、Corr，另一个节点从原NPZ独立重算。TEST此前已查看，不称新盲测、不据此选择结构。\n',encoding='utf8')
files=[(report,'docs/candidate_posttrain_qualification.md')]
for f in source.iterdir():
    if f.suffix in ('.py','.json','.npy'):files.append((f,'work/'+source.name+'/'+f.name))
for name in ('prepare_candidate_final_tools_v1.py','freeze_candidate_posttrain_stage_v1.py','prepare_candidate_pureCPU_runtime_v1.py','seal_candidate_posttrain_preparation_v1.py','seal_recovery_periodic_checkpoint_v1.py','publish_candidate_posttrain_sources_v1.py'):
    files.append((base/'work'/name,'work/'+name))
for name in ('A_candidate_posttrain_native_capture.json','A_candidate_posttrain_native_exit.json','A_candidate_posttrain_native_result.json','B_candidate_posttrain_native_capture.json','B_candidate_posttrain_native_exit.json','B_candidate_posttrain_native_result.json','C_candidate_pureCPU_runtime_capture.json','C_candidate_pureCPU_runtime_exit.json','C_candidate_pureCPU_runtime_result.json','C_pureCPU_readonly_preflight.json','candidate_posttrain_preparation_C_D_verification.json','recovery_checkpoint800_C_D_verification.json'):
    files.append((ev/name,'results/polarity_intensity_continuation/'+name))
manifest=json.loads((repo/'source_manifest.json').read_text(encoding='utf8'));records={r['path']:r for r in manifest['records']};changed=[]
for source,relative in files:
    raw=source.read_bytes();clean,changes=(raw,{}) if source.suffix=='.npy' else sanitize(raw)
    if source.suffix=='.py':ast.parse(clean.decode('utf-8-sig'))
    target=repo/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean)
    records[relative]=dict(path=relative,origin=source.relative_to(base).as_posix(),kind='source' if source.suffix=='.py' else 'research_note' if source.suffix=='.md' else 'generated_order' if source.suffix=='.npy' else 'result',bytes=len(clean),original_sha256=hashlib.sha256(raw).hexdigest(),published_sha256=hashlib.sha256(clean).hexdigest(),sanitizations=changes);changed.append(relative)
manifest['records']=[records[k] for k in sorted(records)];manifest['latest_source_update']='Both-node synthetic candidate posttrain tail qualification and isolated C CPU runtime; final candidate4000/VAL/TEST not yet completed.'
(repo/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');changed.append('source_manifest.json')
readme=repo/'README.md';s=readme.read_text(encoding='utf8');s+='\n[极性／强度完成后评价流程核验](docs/candidate_posttrain_qualification.md)：两节点合成CPU和独立CPU依赖已核验，真实100及新最终VAL／TEST五项尚未完成。\n';readme.write_text(s,encoding='utf8');changed.append('README.md')
for relative in changed:
    data=(repo/relative).read_bytes();oid=command(['hash-object','-w','--stdin'],data).decode().strip();command(['update-index','--add','--cacheinfo','100644',oid,relative]);assert command(['show',':'+relative])==data
command(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit','-m','Qualify saved candidate tails and prepare official evaluation pipeline'])
command(['push','origin','main']);head=command(['rev-parse','HEAD']).decode().strip();assert head==command(['ls-remote','origin','refs/heads/main']).decode().split()[0] and not command(['status','--porcelain']).strip()
receipt=dict(status='SOURCE_RESULTS_EXACT_BLOB_PUSH_VERIFIED',actualclock_before_publication_UTC=a.stamp,commit=head,files=len(changed),no_new_final_five=True)
(ev/'candidate_posttrain_GitHub_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
state_path=base/'outputs/自主优化实际接续.json';state=json.loads(state_path.read_text(encoding='utf8'));state['github_source']=receipt;state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(receipt))

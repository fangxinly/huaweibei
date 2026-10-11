"""Publish authorized current sources and truthful interrupted-state report as exact blobs."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
from prepare_github_source_upload import sanitize
base=Path(__file__).resolve().parents[1];dest=base/'github_upload_20261008T133346Z';ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def command(args,stdin=None):return subprocess.check_output([git]+args,cwd=dest,input=stdin)
p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);a=p.parse_args()
report=base/'outputs/极性强度匹配两臂实际续训.md'
report.write_text('# 原流极性与强度升级：实际训练与中断\n\n保存记录时钟：'+a.stamp+'。\n\n两臂均已完成真实前16步，完整模型、345个Adam状态、调度器、随机状态和订单已上传Release，并在另一服务器从公开原件下载核验。极性×强度头与辅助回归对照额外参数均为2002，输入为全部模态；相同初始模型状态与订单。恢复前16后实际开始100轮/4000更新，官方VAL批MSE严格earliest选模。\n\n最新真实观察：A进程2432停在第97轮/3880步，历史best73 VAL选模MSE0.683284574；B进程2878停在第96轮/3840步，历史best92 0.720681450。进程已不在，缺少自然退出和完成capture，原因尚未建立。这不是100轮完成，也不是最终五项成绩；选模MSE不能冒充MAE。\n\n第400步完整可恢复状态均保留，当前完整中断原件正在新根封存；修复后的400→440步合成CPU恢复测试在两台机器、两种模式均自然0通过，模型参数/预测/Adam/调度器/随机状态一致。首测试的步数断言笔误失败原件保留。后续恢复将脱离SSH会话运行，并每10轮保存原子可变检查点；旧冻结原件不删除或覆盖。恢复会重放检查点之后丢失的计算，额外计算成本必须计入记录，不能算新独立重复实验。\n\n方法与订单、辅助损失、100轮逻辑预算、VAL选模规则保持原冻结方案。它是普通监督结构升级，尚不是折外残差效用控制，仍存在global gain校准捷径，不冒共享/互补语义证明。\n\n完整100最终自然0与保存核验通过后，分别计算两臂选定checkpoint的官方VAL和TEST全部五项，并列CaReFlow。TEST已查看，不能称新盲测，不用于选择结构。\n\n已完成MSE-only方案VAL MAE0.608080、TEST0.652712；CaReFlow TEST0.619535，新MSE五项均落后。旧原流的约0.59来自VAL（0.597257），其TEST是0.650616。详[已完成五项结果](autonomous_mse_training.md)。\n',encoding='utf8')
files=[(report,'docs/polarity_intensity_training.md')]
names=['prepare_polarity_intensity_resume_v2.py','audit_public_prefix16_candidate_v2.py','run_candidate_peer_audit_capture_v2.py','freeze_candidate_peer_prefix_audit_v2.py','freeze_candidate_resume100_v2.py','launch_remote_release_stream_node_v2.py','prepare_polarity_intensity_publication_v2.py','freeze_polarity_intensity_prefix_v2.py','capture_interrupted_candidate_v1.py','prepare_candidate_recovery400_v1.py','audit_interrupted_candidate400_v1.py','seal_candidate_recovery_native_v1.py','publish_candidate_continuation_sources_v1.py']
files += [(base/'work'/n,'work/'+n) for n in names]
for folder in ('polarity_intensity_resume_preparation_20261008T205030Z','polarity_intensity_factorized_aux_resume100_qualified_20261008T205936Z','polarity_intensity_regression_aux_resume100_qualified_20261008T205936Z','candidate_recovery400_preparation_20261009T000700Z','candidate_recovery400_preparation_20261009T000900Z'):
 files += [(f,'work/'+folder+'/'+f.name) for f in (base/'work'/folder).iterdir() if f.suffix in ('.py','.json','.npy')]
files += [(f,'results/polarity_intensity_continuation/'+f.name) for f in ev.glob('*.json')]
manifest=json.loads((dest/'source_manifest.json').read_text(encoding='utf8'));records={r['path']:r for r in manifest['records']};changed=[]
for source,relative in files:
 raw=source.read_bytes();clean,changes=(raw,{}) if source.suffix=='.npy' else sanitize(raw)
 if source.suffix=='.py':ast.parse(clean.decode('utf-8-sig'))
 target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean)
 records[relative]=dict(path=relative,origin=source.relative_to(base).as_posix(),kind='source' if source.suffix=='.py' else 'research_note' if source.suffix=='.md' else 'generated_order' if source.suffix=='.npy' else 'result',bytes=len(clean),original_sha256=hashlib.sha256(raw).hexdigest(),published_sha256=hashlib.sha256(clean).hexdigest(),sanitizations=changes);changed.append(relative)
manifest['records']=[records[k] for k in sorted(records)];manifest['latest_source_update']='Matched polarity intensity actual prefixes preserved/peer verified; real continuation interrupted without natural exit at97/96; complete400 recovery qualified on synthetic CPU, no new final VAL/TEST five scores.'
(dest/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');changed.append('source_manifest.json')
readme=dest/'README.md';s=readme.read_text(encoding='utf-8-sig');s+='\n\n## 原流极性与强度升级\n\n[真实两臂续训、中断及恢复记录](docs/polarity_intensity_training.md)。完整prefix16已保存并跨服务器CPU通过；100续训在97/96轮中断，尚无新最终VAL/TEST五项，完整400状态恢复资格已核验，禁止将选模MSE当MAE。\n';readme.write_text(s,encoding='utf8');changed.append('README.md')
for rel in changed:
 data=(dest/rel).read_bytes();oid=command(['hash-object','-w','--stdin'],data).decode().strip();command(['update-index','--add','--cacheinfo','100644',oid,rel]);assert command(['show',':'+rel])==data
command(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit','-m','Preserve matched flow candidate prefixes and qualify interrupted checkpoint recovery'])
command(['push','origin','main']);head=command(['rev-parse','HEAD']).decode().strip();assert head==command(['ls-remote','origin','refs/heads/main']).decode().split()[0] and not command(['status','--porcelain']).strip()
(ev/'candidate_continuation_GitHub_receipt.json').write_text(json.dumps(dict(status='SOURCE_RESULTS_EXACT_BLOB_PUSH_VERIFIED',actualclock_before_publication_UTC=a.stamp,commit=head,files=len(changed),no_new_final_five=True),indent=2),encoding='utf8');print(json.dumps(dict(commit=head,files=len(changed))))

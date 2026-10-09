"""Publish pinned recovery source and actual status without changing frozen originals."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
from prepare_github_source_upload import sanitize
base=Path(__file__).resolve().parents[1];dest=base/'github_upload_20261008T133346Z';ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def command(args,stdin=None):return subprocess.check_output([git]+args,cwd=dest,input=stdin)
p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);a=p.parse_args()
state=json.loads((base/'outputs/自主优化实际接续.json').read_text(encoding='utf8'));pair=state['polarity_intensity_pair_v2']['recovery400_actual'];running=state['status']=='MATCHED_PAIR_DETACHED_RECOVERY400_RUNNING'
report=base/'outputs/极性强度匹配两臂实际续训.md'
report.write_text('# 原流极性与强度升级：实际训练及恢复\n\n记录时钟：'+a.stamp+'。\n\n两臂真实前16步完整状态已发布Release，并由另一台服务器从公开原件下载核验。极性×强度头与辅助回归对照均增加2002参数、使用全部模态，相同初始参数和订单。\n\n原100轮进程在第97/96轮中断，没有自然退出回执，原因尚未建立。最后落盘更新分别3893/3873。完整原件在新根封存，原SHA、ZIP CRC、唯一成员及全部成员SHA通过；各1.76GB原ZIP已直传GitHub Release两分片。原400步完整模型、345个Adam状态、调度器、RNG、订单及前10轮选模均保留，旧冻结原件未删除或修改。\n\n400→440步合成恢复在两台服务器、两种模式均自然0通过，参数、预测、Adam、scheduler及随机状态一致；首断言笔误失败证据保留。'+('跨服务器公开下载完整原件、400步原状态CPU核验均自然0通过；两臂已经实际脱离SSH会话从400步完整恢复，至少到11轮，恢复轨迹参数和预测SHA与中断前记录一致。每10轮原子替换一个明确可变的完整恢复检查点。' if running else '跨服务器完整原件下载和400步原状态CPU核验正在运行；尚未启动真实恢复。')+'\n\n恢复需重放A3493/B3473条已记录更新，额外计算成本明确单列，未落盘最后更新仍不能排除。最终逻辑训练为100轮4000更新，原400步计一次；这不是新的独立重复实验，实际累计计算预算会高于无中断的4000步。方法、固定损失、共同订单与官方VAL选模规则均保持原冻结方案。\n\n当前没有新方案最终VAL/TEST五项，历史选模MSE不能冒充MAE。完整100自然0后先保存原状态并跨服务器CPU审核，再固定同一选定checkpoint计算官方VAL229和TEST685全部五项，并列CaReFlow。TEST已经查看，不称新盲测，不按TEST选结构。\n\n已完成MSE-only方案VAL MAE0.608080、TEST0.652712，CaReFlow TEST0.619535，新MSE五项均落后。旧原流约0.59来自VAL0.597257，TEST为0.650616。见[已完成五项](autonomous_mse_training.md)。极性强度结构是普通监督升级，尚不是折外残差效用控制，不冒共享/互补语义证明。\n',encoding='utf8')
files=[(report,'docs/polarity_intensity_training.md')]
for n in ('run_interrupted_peer400_capture_v1.py','freeze_interrupted_peer400_v1.py','freeze_candidate_recovery400_v1.py','dispatch_qualified_detached_v1.py','observe_capture_short_v1.py','update_pair_recovery_status_v1.py','publish_recovery_status_update_v1.py','prepare_peer400_seekfix_v1.py','seal_interrupted_peer400_v1.py','seal_recovery_actual_start_v1.py'):
 files.append((base/'work'/n,'work/'+n))
for folder in (base/'work').iterdir():
 if folder.is_dir() and any(folder.name.startswith(s) for s in ('A_interrupted_peer400_qualified_','B_interrupted_peer400_qualified_','A_interrupted_peer400_seekfix_qualified_','B_interrupted_peer400_seekfix_qualified_','A_candidate_recovery400_qualified_','B_candidate_recovery400_qualified_')):
  files += [(f,'work/'+folder.name+'/'+f.name) for f in folder.iterdir() if f.suffix in ('.py','.json','.npy')]
files += [(f,'results/polarity_intensity_continuation/'+f.name) for f in ev.glob('*.json')]
manifest=json.loads((dest/'source_manifest.json').read_text(encoding='utf8'));records={r['path']:r for r in manifest['records']};changed=[]
for source,relative in files:
 raw=source.read_bytes();clean,changes=(raw,{}) if source.suffix=='.npy' else sanitize(raw)
 if source.suffix=='.py':ast.parse(clean.decode('utf-8-sig'))
 target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean)
 records[relative]=dict(path=relative,origin=source.relative_to(base).as_posix(),kind='source' if source.suffix=='.py' else 'research_note' if source.suffix=='.md' else 'generated_order' if source.suffix=='.npy' else 'result',bytes=len(clean),original_sha256=hashlib.sha256(raw).hexdigest(),published_sha256=hashlib.sha256(clean).hexdigest(),sanitizations=changes);changed.append(relative)
manifest['records']=[records[k] for k in sorted(records)];manifest['latest_source_update']='Matched fixed flow recovery: original interrupted archives Release preserved; actual status '+state['status']+'; replay cost separately recorded; no new final five.'
(dest/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');changed.append('source_manifest.json')
readme=dest/'README.md';s=readme.read_text(encoding='utf-8-sig');start=s.find('## 原流极性与强度升级')
assert start>=0;s=s[:start]+'## 原流极性与强度升级\n\n[实际训练、完整中断保存及恢复](docs/polarity_intensity_training.md)。当前状态：'+state['status']+'。尚无新最终VAL/TEST五项，恢复额外计算成本单列。\n';readme.write_text(s,encoding='utf8');changed.append('README.md')
for rel in changed:
 data=(dest/rel).read_bytes();oid=command(['hash-object','-w','--stdin'],data).decode().strip();command(['update-index','--add','--cacheinfo','100644',oid,rel]);assert command(['show',':'+rel])==data
command(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit','-m','Preserve interrupted originals and qualify detached fixed candidate recovery'])
command(['push','origin','main']);head=command(['rev-parse','HEAD']).decode().strip();assert head==command(['ls-remote','origin','refs/heads/main']).decode().split()[0] and not command(['status','--porcelain']).strip()
(ev/'candidate_continuation_GitHub_receipt.json').write_text(json.dumps(dict(status='SOURCE_RESULTS_EXACT_BLOB_PUSH_VERIFIED',actualclock_before_publication_UTC=a.stamp,commit=head,files=len(changed),no_new_final_five=True),indent=2),encoding='utf8');print(json.dumps(dict(commit=head,files=len(changed))))

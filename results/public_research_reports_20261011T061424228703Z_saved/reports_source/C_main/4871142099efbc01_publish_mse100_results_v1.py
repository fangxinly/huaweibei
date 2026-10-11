"""Publish verified official scores and qualification source copies, no retraining."""
import ast,hashlib,json,subprocess
from pathlib import Path
from prepare_github_source_upload import sanitize

base=Path(__file__).resolve().parents[1];dest=base/'github_upload_20261008T133346Z';ev=base/'outputs/autonomous_mse16_20261008T175549Z'
git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def command(args,stdin=None):return subprocess.check_output([git]+args,cwd=dest,input=stdin)
# A prior publication preparation may have written these same authorized copies.
proof=json.loads((ev/'A100_independent_proofs_publication.json').read_text(encoding='utf8'));assert proof['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
manifest=json.loads((dest/'source_manifest.json').read_text(encoding='utf8'));records={r['path']:r for r in manifest['records']}
files=[(base/'outputs/MSE100官方VAL_TEST实际结果.md','docs/autonomous_mse_training.md')]
names=['A100_official_five_result.json','B100_audit_result.json','A100_inference_result.json','A100_score_publication.json','B100_independent_five_result.json','B100_independent_five_capture.json','B100_independent_five_exit.json','B100_independent_five_C_verification.json','A100_independent_proofs_publication.json','MSE100_five_D_small_verification.json','B100_score_first_audit_capture.json','B100_score_first_audit_exit.json']
files += [(ev/n,'results/autonomous_mse100/'+n) for n in names]
files += [(p,'work/official_mse_posttrain_tools_v1/'+p.name) for p in (base/'work/official_mse_posttrain_tools_v1').glob('*.py')]
for n in ('seal_mse100_official_results_v1.py','prepare_polarity_intensity_prefix_v2.py','run_candidate_native_capture_v1.py','run_small_original_CPU_audit_capture_v1.py','publish_mse100_results_v1.py'):
 files.append((base/'work'/n,'work/'+n))
files += [(p,'work/polarity_intensity_prefix_v2_20261008T203225Z/'+p.name) for p in (base/'work/polarity_intensity_prefix_v2_20261008T203225Z').iterdir() if p.suffix in ('.py','.npy','.json')]
changed=[]
for source,relative in files:
 raw=source.read_bytes();clean,changes=(raw,{}) if source.suffix=='.npy' else sanitize(raw)
 if source.suffix=='.py':ast.parse(clean.decode('utf-8-sig'))
 target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean)
 records[relative]=dict(path=relative,origin=source.relative_to(base).as_posix(),kind='research_note' if source.suffix=='.md' else 'source' if source.suffix=='.py' else 'result' if source.suffix=='.json' else 'generated_order',bytes=len(clean),original_sha256=hashlib.sha256(raw).hexdigest(),published_sha256=hashlib.sha256(clean).hexdigest(),sanitizations=changes)
 changed.append(relative)
manifest['records']=[records[k] for k in sorted(records)];manifest['latest_source_update']='MSE100 official VAL/TEST scored and independently verified from frozen NPZ; negative result. Polarity/intensity matched pair native qualification in progress, no real candidate TRAIN yet.'
(dest/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');changed.append('source_manifest.json')
readme=dest/'README.md';s=readme.read_text(encoding='utf-8-sig');s+='\n\n## 最新完整MSE实验\n\n[官方VAL/TEST全五项、原状态与独立核验](docs/autonomous_mse_training.md)：100轮/4000更新完成，VAL选65；VAL MAE0.608080，TEST0.652712，未超过CaReFlow TEST0.619535。B从冻结原NPZ独立重算最大差异5.55e-16；完整原状态及失败核验原件均保留。[Release](https://github.com/fangxinly/huaweibei/releases/tag/autonomous-flow-health-20261008)。新极性/强度匹配两臂源码正在原环境资格核验，尚无真实性能结果。\n';readme.write_text(s,encoding='utf8');changed.append('README.md')
for relative in changed:
 data=(dest/relative).read_bytes();oid=command(['hash-object','-w','--stdin'],data).decode().strip();command(['update-index','--add','--cacheinfo','100644',oid,relative]);assert command(['show',':'+relative])==data
command(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit','-m','Report verified MSE100 official scores and prepare matched polarity intensity prefix'])
command(['push','origin','main']);head=command(['rev-parse','HEAD']).decode().strip();remote=command(['ls-remote','origin','refs/heads/main']).decode().split()[0];assert head==remote and not command(['status','--porcelain']).strip()
(ev/'MSE100_results_GitHub_receipt.json').write_text(json.dumps(dict(status='RESULT_SOURCE_PUBLICATION_PUSH_VERIFIED',commit=head,remote_head=remote,files=len(changed),new_real_candidate_training=False),indent=2),encoding='utf8');print(json.dumps(dict(commit=head,files=len(changed))))

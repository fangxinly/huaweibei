import json,hashlib,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
utc=datetime.now(timezone.utc).isoformat()
SRC=ROOT/'work/locked_T0_constant_outer_development_20261006T1324Z'
D=Path(json.loads((SRC/'permanent_D_location.json').read_text())['directory'])
ledger_path=ROOT/'outputs/研究建议交流接续.json';ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
for n in ('研究建议交流接续.json','研究接续状态.md'):
 p=D/'preceding_state'/n;p.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'outputs'/n,p)
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第八次独立审视_结构化收益条件与最小匹配对照.md')
if not any(x.get('sha256')==sha(review) for x in ledger['received_reviews']):
 ledger['received_reviews'].append({'path':str(review),'sha256':sha(review),'read_and_considered':True,'read_recorded_utc':utc,'turn':'01a11156-3c19-74a3-bec8-3a1388ee0db5','accepted':'Two U/W training arms; fixed free/interval/discrete readouts; shared information with delta; global train-video normalization with fixed penalty and zero-weight fallback; asset label graph before experiments.','scope_correction':'Old two-sample weighted fit witness not acceptance gain and not covered by independent array auditor; symbolic regret proof not established by floating witnesses.'})
ledger['review_thread'].update({'status':'ninth_actual_locked_constant_in_progress_after_one_snapshot','last_completed_turn':'01a11156-3c19-74a3-bec8-3a1388ee0db5','current_turn':'01a11170-52b3-7b01-9299-4d3d7a6f81ed','last_wait_cursor':'5799b749-1b95-49b8-8e5d-e5879f2149c3:1'})
ledger['squared_risk_residual_math_decision']['review_status']='EIGHTH_COMPLETE_FULL_REPORT_READ_AND_QUALIFIED'
ledger['locked_T0_constant_outer_development']={'batch':'locked_T0_constant_outer_development_actual_20261006T133640Z','phase':'ACTUAL_LOCAL_CPU_DEVELOPMENT_NO_REFIT','permanent_D':str(D),'report':str(ROOT/'outputs/同T0锁定常数外折开发测量实际结果.md'),'plan_sha256':sha(SRC/'plan.json'),'prediction_sha256':sha(SRC/'execute/locked_predictions.npz'),'rows':433,'videos':18,'video_mse':[.5517555139650273,.5344150922693057],'improved_videos':9,'predeclared_required_improved_videos':12,'cost_heuristic_passed':False,'decision':'STOP_THIS_SCALAR_BRANCH_EXPANSION_KEEP_HISTORICAL_DEVELOPMENT_BASELINE_NO_RESCUE_SWEEPS','sent_to_review':True,'new_confirmation':False,'whole_pipeline_crossfit':False,'new_fit':False,'new_GPU':False,'other_node_CPU':False,'new_remote_capture':False}
batch='locked_T0_constant_outer_development_actual_20261006T133640Z'
if not any(x.get('batch')==batch for x in ledger['evidence_batches']):
 ledger['evidence_batches'].append({'batch':batch,'status':'sent','report':str(ROOT/'outputs/同T0锁定常数外折开发测量实际结果.md'),'report_sha256':sha(ROOT/'outputs/同T0锁定常数外折开发测量实际结果.md'),'prompt_path':str(D/'review_send_prompt.json'),'repeat_send_allowed':False,'actual_local_CPU_experiment':True,'actual_GPU':False,'actual_other_node_CPU':False})
new_path=ROOT/'outputs/新三P4第二租期实际接入核验_20261006T1337Z.json'
new=json.loads(new_path.read_text(encoding='utf-8'))
ledger['second_lease_new_assets']={'human_new_allocation_authorized':True,'record':str(new_path),'record_sha256':sha(new_path),'nodes':new['nodes'],'human_lease_source':new['lease_source'],'estimated_expiry_utc':new['estimated_expiry_utc'],'platform_lease_verified':False,'old_allocation_endpoints_forbidden':True,'asset_restore_pending':True,'new_GPU_precheck':False,'new_training':False}
ledger['scientific_state']='锁定T0常数OUTER433/18本地CPU开发测量降3.14%但9/18低于12/18门槛，停止标量扩张；第八完整读完，第九结果审视进行。人类新三P4 24h已认证独立UUID/空compute，干净环境需恢复公共资产，新科学GPU预检/100未执行。'
ledger['updated_at_utc']=utc;ledger['updated_utc']=utc
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=ROOT/'outputs/研究接续状态.md';text=state.read_text(encoding='utf-8').splitlines()
out=[]
for line in text:
 if line.startswith('更新UTC '):line=f'更新UTC {utc}。以人类最新新P4授权及短接续为准；不重注入历史。研究与新旧租期保存未整体完成。'
 elif line.startswith('1) '):
  line='1) 最新真实研究：同T0锁定原FIT常数c=-.07411910583740507，在原OUTER433/18开发测量，无重拟合/扫参。视频MSE.551755514→.534415092降3.14%，9/18改善低于预声明12/18，平均Q负且删一视频仍负；停止本标量扩张，不因平均正收益启动U/W/Q/消息100。plan f4a1a2d48d3dec88a5fc4850840be5a448f1fdd5c01839511eccbbf57044f1a0，预测SHA后只433 OUTER y，无混折mu/DEV/TEST，旧全TRAIN探索/旧CAL-EVAL行重叠不抹掉，非新确认。D locked_T0_constant_outer_development_actual_20261006T133640Z 18原件SHA/ZIPCRC/独立本地CPU审核过；不是GPU/异节点CPU/remote capture。旧同T0 FIT-INNER A2.56%、B失败和原D证据保留。'
 elif line.startswith('4) '):line='4) 前一租期最新三remote capture仍UTC11:33/11:39，D lease_final_window_20261006T113906Z全SHA/CRC/旧科学/大引用核过；完整teacher full/原异节点CPUfresh联结保留，大SHA不是新下载，UTC08:08/10:08未执行不回填。旧估Oct6UTC12:08:17平台未核，不冒释放/重建。人类已给新三P4与24h，旧余量失败不延伸禁止新资产；新执行仍先真实预算及保存门控。'
 elif line.startswith('5) '):line='5) 旧A key变化/B凭据拒绝及C12:06只读exit0历史保留，旧run路径空不冒新完成；所有旧SSH/SFTP已关闭禁复用。新节点接入见新三P4第二租期接续.md：13:37:40三独立P4/空compute，/data约47.18GB，旧root不存在；Torch2.1.0+cu121/NumPy1.26.4，Transformers缺。开始公共资产恢复，未实际科学GPU预检。新SSH79535/62639/17537当前仍活动，关闭后核exit0并禁复用；first-use key接受A/B写known_hosts失败/C成功，未手工改校验。'
 elif line.startswith('8) '):line='8) 第六原usage失败保留，续第六/第七/第八完整读完；第八采纳两U/W训练臂与固定读出、权重归一和角色核查，补充旧合成审核范围。第九锁定常数结果批D保存后已发一次，turn01a11170-52b3-7b01-9299-4d3d7a6f81ed仍inProgress/cursor1；按交流JSON一次read/wait，不冒返回。仅同建议聊天01a10fcb-6663-70a2-9a76-60e5634d0c03人类授权双向，分析无GPU/改主源/训练/子代理，不发凭据/新建聊天。健康/小准备/同证据不重复发。'
 elif line.startswith('9) '):line='9) 仅这批新A REDACTED_SERVER_HOST.invalid:53314 GPU-53696803-875e-eec8-2231-29db63579891；B REDACTED_SERVER_HOST.invalid:53332 GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f；C REDACTED_SERVER_HOST.invalid:53517 GPU-417d3577-0525-788b-7296-0808a0f52012。旧全部地址/N/R永禁连。新租期人类24h，约Oct7UTC13:35/BJ21:35估，平台未核，不把JSON冒来源。fresh真实password只本聊天人类新凭据，禁密码文件/命令/自动/猜测；先UUID/完整argv/compute/源/资产/空间，结束exit/bye核实际结果。拒绝不换路绕过。'
 out.append(line)
state.write_text('\n'.join(out)+'\n',encoding='utf-8')
print(json.dumps({'status':'SHORT_STATE_LEDGER_NEW_HUMAN_LEASE_AND_LOCKED_DEVELOPMENT_UPDATED','actual_utc':utc,'new_GPU_precheck':False,'review_ninth_complete':False}))

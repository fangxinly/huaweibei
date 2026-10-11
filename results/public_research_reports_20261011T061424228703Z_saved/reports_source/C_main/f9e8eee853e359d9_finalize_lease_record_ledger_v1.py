from pathlib import Path
import argparse,datetime,json,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
o=Path(__file__).resolve().parents[1]/'outputs';r=a.directory
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ledger=read(o/'研究建议交流接续.json');auto=read(r/'official_automation_update.json')
assert not auto['result'].get('isError',False)
send=read(r/'review_decision_resume_receipt.json');assert not send['result'].get('isError',False)
snapshot=read(r/'review_resume_status_snapshot.json')
poll=json.loads(snapshot['content'][0]['text'])['polls'][0]
ledger['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
ledger['automation'].update(updated_at_utc=auto['utc'],prompt_sha256=hashlib.sha256(auto['prompt'].encode()).hexdigest(),result='EXISTING_AUTOMATION_OFFICIAL_UPDATE_SUCCESS_ACTIVE',original_tool_receipt=str(r/'official_automation_update.json'))
ledger['review_thread'].update(status='sixth_original_usage_failure_preserved_new_decision_review_inProgress_not_completed',current_turn=poll['latestTurn']['id'],last_wait_cursor=poll['cursor'])
ledger['evidence_batches'].append({'batch':send['batch'],'status':'sent_once_new_turn_inProgress','sent_utc':send['utc'],'prompt_receipt':str(r/'review_decision_resume_receipt.json'),'prompt_receipt_sha256':sha(r/'review_decision_resume_receipt.json'),'report':str(o/'视频隔离完整流参考最小目标修订草案_v2.md'),'report_sha256':sha(o/'视频隔离完整流参考最小目标修订草案_v2.md'),'repeat_send_allowed':False,'scientific_result':False})
ledger['scientific_state']+=' 原第六用量失败保留；新最小目标v2与cheap残差决策批已一次续接，当前只有commentary，未完成。'
ledger['latest_research_records_preservation']={'directory':str(r/'research_records'),'receipt_sha256':sha(r/'research_records/preservation_receipt.json'),'remote_capture_or_CPU_forward':False}
(o/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
state=(o/'研究接续状态.md').read_text(encoding='utf-8')
state+='\n补充UTC '+ledger['updated_at_utc']+'：原第六usage失败后，新v2目标/cheap残差决策批已发送一次，turn '+poll['latestTurn']['id']+'，cursor '+poll['cursor']+' 当前inProgress，仅commentary未完成。初步指出cycle沿未detach状态/context影响前向网络、可先试同折标量常数/仿射；只是建议，具体协议/真实CPU拟合未执行。下轮一次read/wait后全文审阅，不把开始回复当完整建议，不重复发同批。当前heartbeat已正式更新ACTIVE10min；新研究记录D保存非远端GPU/CPU前向。\n'
(o/'研究接续状态.md').write_text(state,encoding='utf-8')
proof={'status':'LEASE_PRESERVATION_AND_RESEARCH_RECORDS_COMPLETE_FOR_THIS_TURN_NOT_WHOLE_RESEARCH',
 'utc':ledger['updated_at_utc'],'latest_capture_audit_sha256':sha(r/'joint_capture_audit.json'),
 'late_to_final_increment_audit_sha256':sha(r/'late_to_final_increment_audit.json'),
 'session_closure_sha256':sha(r/'session_closure.json'),'official_automation_receipt_sha256':sha(r/'official_automation_update.json'),
 'research_records_first_sha256':sha(r/'research_records/preservation_receipt.json'),
 'review_resume_receipt_sha256':sha(r/'review_decision_resume_receipt.json'),
 'historical_capture_gaps':['UTC08:08','UTC10:08'],'new_GPU_experiment_started':False,'new_CPU_residual_fit_started':False,
 'full_research_complete':False,'expiry_platform_verified':False}
(r/'turn_completion_receipt.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__),r/Path(__file__).name)
print(json.dumps(proof))

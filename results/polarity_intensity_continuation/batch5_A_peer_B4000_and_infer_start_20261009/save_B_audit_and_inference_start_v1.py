import datetime,hashlib,json,os,pathlib,shutil,tomllib,zipfile
retry=pathlib.Path(__file__).resolve().parent;ev=retry.parent;prep=ev/'batch5_B_inference_manifest_repair_prepared_20261009T174820Z';failed=ev/'batch5_B_inference_prepared_20261009T174412Z';out=pathlib.Path('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs');prior=ev.parent/'candidate_posttrain_lowC_20261009T005229Z';current=prior/'D_current_research_state.json';root=ev/'batch5_B_audit_and_inference_started_saved_actual_20261009T175221Z'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_bytes(raw(v))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert shutil.disk_usage('D:/').free>80*1024**2 and shutil.disk_usage('C:/').free>205*1024**2;root.mkdir(exist_ok=True)
for parent,names in [(retry,['N1_D_preservation_receipt.json','N1_Release_publication_receipt.json','N2_D_preservation_receipt.json','N2_Release_publication_receipt.json','N2_complete_actual_small_original.zip','N2_poll_174140_original.json','N2_poll_174140_tool_original.json','N3_identity_original.json']), (prep,['batch5_B_posttrain_v1.py','official_new_N2_inference_protocol.json','prepared_source_D_receipt.json','restore_dispatch_original.json','inference_dispatch_original.json','inference_dispatch_tool_original.json','runtime_D_preservation_receipt.json','complete_public_runtime_restore_original.zip','verify_runtime_original_v1.py','posttrain_manifest_repair_source.zip']), (failed,['first_public_restore_failure_D_receipt.json','first_public_restore_failure_original.zip','restore_failure_read_original.json','restore_failure_read_tool_original.json','prepare_manifest_repair_v2.py'])]:
 dest=root/({retry:'original_retry',prep:'original_runtime',failed:'original_failure'}[parent]);dest.mkdir(exist_ok=True)
 for n in names:
  if (dest/n).exists():assert sha(dest/n)==sha(parent/n)
  else:shutil.copyfile(parent/n,dest/n)
shutil.copyfile(__file__,root/pathlib.Path(__file__).name)
a=read(retry/'N1_D_preservation_receipt.json');b=read(retry/'N2_D_preservation_receipt.json');publication=read(retry/'N2_Release_publication_receipt.json');runtime=read(prep/'runtime_D_preservation_receipt.json');dispatch=read(prep/'inference_dispatch_original.json')
assert b['result']['all_Adam_steps']==4000 and b['result']['best_epoch']==92 and publication['remote_digest_verified'] and publication['source_SHA']==b['archive_SHA']
assert runtime['natural_exit']['natural_exit']==0 and dispatch['B_selected92_same_checkpoint']
entry=dict(status='B_FULL4000_D_AND_RELEASE_PASSED_ORIGINAL_B_INFERENCE_ACTUALLY_DISPATCHED',actual_UTC=utc(),A_other_node_five_complete=a['result'],B_full4000=b,B_full4000_Release=publication,new_N2_runtime=runtime,first_asset_manifest_guard_failure=read(failed/'first_public_restore_failure_D_receipt.json'),B_official_inference_dispatch=dispatch,B_official_inference_result_pending=True,B_five_not_yet_computed=True,original_A_author_score_not_repeated=True,old_batch4_endpoints_permanently_forbidden=True,extra_replay_cost_A3493_B3473_separate=True)
write(root/'actual_research_current.json',entry);write(root/'A_other_node_five_result_original.json',a['result']);write(root/'B_full4000_result_original.json',b['result'])
report='A的原NPZ/官方ID标签异节点VAL与TEST五项已独立复核通过，最大差异1.5543122344752192e-15；原作者一次评分未重做。A仍仅TEST Acc2/F1优于CaReFlow，Acc7/MAE/Corr落后，未五项整体超过。\n\nB regression_aux原100轮/4000逻辑更新的异节点CPU审核于UTC17:38:58通过，原core child539自然0UTC17:39:07，345Adam全4000、scheduler/RNG/100轮顺序及strict earliest选92核过，CPU最大差异2.8312206268310547e-6。原完整训练包29ff0e21…及三片/全成员SHA/CRC唯一通过，审核小原ZIP aa9876da…已经D和Release保存。额外记录重放A3493/B3473仍单列，非独立重复。\n\n新N2隔离环境原24版本/官方资产SHA/两模式原CandidateTail合成资格通过。首公共包自清单未计入预期成员导致恢复预检自然1，原失败包616c7556…完整保留；只修清单处理，原模型、infer核心、loss/选模规则未改。原B固定选92零标签VAL229/TEST685推理wrapper1028 UTC17:52:26已实际启动；尚无完成预测、字节审计或新五项，不能把启动当完成。旧到期batch4永禁连，新租期按人类24h保守Oct10UTC16:30:58/第三16:47:25，不称平台确认。\n'
(root/'研究实际进展.md').write_text(report,encoding='utf8')
files=sorted(p for p in root.rglob('*') if p.is_file());cap=root/'complete_B_audit_runtime_failure_repair_and_infer_start_original.zip';rows=[]
with zipfile.ZipFile(cap,'x',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  d=p.read_bytes();n=p.relative_to(root).as_posix();z.writestr(n,d);rows.append(dict(name=n,bytes=len(d),sha256=hashlib.sha256(d).hexdigest()))
 z.writestr('member_manifest.json',raw(rows))
with zipfile.ZipFile(cap) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in rows}|{'member_manifest.json'}
 for v in rows:assert len(z.read(v['name']))==v['bytes'] and hashlib.sha256(z.read(v['name'])).hexdigest()==v['sha256']
receipt=dict(actual_UTC=utc(),archive=str(cap),archive_SHA=sha(cap),archive_bytes=cap.stat().st_size,member_count=len(rows),all_member_SHA_CRC_unique_passed=True,no_new_remote_inference_completion_claim=True)
write(root/'D_preservation_receipt.json',receipt)
s=read(current);s['status']='A_FIVE_OTHER_NODE_VERIFIED_B_FULL4000_PASSED_B_OFFICIAL_INFERENCE_ACTUALLY_STARTED';s['latest_B_full4000_actual']=dict(D=b,Release=publication);s['latest_B_official_inference_actual']=dict(dispatch=dispatch,pending_result=True,new_N2_runtime_D=runtime,source_preparation=str(prep),first_failure_original=str(failed/'first_public_restore_failure_D_receipt.json'))
s['latest_batch5_actual_current']=dict(root=str(root),evidence=entry,D_preservation_receipt=receipt);s['latest_local_A_five_actual']['B_CPU4000_not_passed']=False;s['latest_local_A_five_actual']['B_full4000_current_reference']=str(retry/'N2_D_preservation_receipt.json');s['latest_batch5_retry_actual']['evidence']['B_full4000_not_yet_passed']=False;s['latest_batch5_retry_actual']['evidence']['B_full4000_current_reference']=str(retry/'N2_D_preservation_receipt.json')
for dest in [current,out/'自主优化实际接续.json']:
 tmp=dest.with_name(dest.name+'.B_started.tmp');tmp.write_bytes(raw(s));os.replace(tmp,dest)
assert current.read_bytes()==(out/'自主优化实际接续.json').read_bytes();write(root/'D_C_current_sync_receipt.json',dict(actual_UTC=utc(),same_SHA=sha(current)))
with (out/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+utc()+' 最新B CPU4000审核自然0且完整D/Release过，A异节点五项通过；新N2原24runtime/资产/合成资格自然0。首公共包清单断言失败原件保留，只修manifest后原B选92官方零标签推理1028 UTC17:52:26实际dispatch，root '+dispatch['root']+'，尚无完成预测/字节审计/新五项。最新完整D包'+receipt['archive_SHA']+'，root '+str(root)+'。\n')
auto=tomllib.loads(pathlib.Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf8'))
prefix=('最新Oct9实际UTC17:52:26接续优先：A异节点N1五项完整D/Release通过不重做作者计分；B原CPU core539 UTC17:38:58审核结果过/17:39:07自然0，wrapper350/child354均自然0已结束禁复用。345Adam全4000/scheduler/RNG/order100/strict earliest选92/CPUflow max2.8312206268310547e-6通过，selectedSHA323f474e6b50da98a9bfd85d515fefa5db4a618f90a8cc246852ccb31b012833。原审核小ZIPaa9876da9eba95ad8e9eb5a0fda883933d806e81b861aabe4ca9c659229ff00a已真实D全member SHA/CRC/unique和Release远端digest通过，原大ZIP永久保留，仅新临时PT核SHA且CPU过后移除。\n'
'新N2隔离cu121原24依赖/官方资产SHA/原两模式CandidateTail合成资格自然0，runtime恢复723 UTC17:51:18自然0，完整原ZIPfda3801a0821c947d62a32b7731cd64d037946deb2f76137d299541eaaedd64e已D全成员核过。首恢复691因原public ZIP自清单未计入预期集合断言自然1，完整失败ZIP616c7556225115a2109092ab921f7a59384ddc9cb8d42a148ae48041714c8367已D全过保留；只修自清单处理并验证原包全成员，新helper683a457d77c61effb9c0708aa69407429689bb5c0f078b507057a298ea9fbec2，原infer/core/模型/loss/选模/资产SHA不改。新protocol fce90632ac66c5b440b7671a020d02e9931e35e3ed0dd657154d399822861c68明确新N2身份非旧B/C。\n'
'B官方同选92零标签VAL229/TEST685原infer已wrapper1028 UTC17:52:26启动，root '+dispatch['root']+'，plan '+dispatch['fullargv'][6]+'。先读natural_exit/capture_receipt/out/inference_result/原child exit/stderr，不重复启动健康任务或消费once。完整预测先D/Release保存，再异节点原数组字节核验，再作者五项一次和另一节点原NPZ独立复核。当前还无B推理完成/五项，不冒成功；A作者一次永不重做。最新完整D保存root '+str(root)+' 包SHA'+receipt['archive_SHA']+' 全member SHA/CRC/unique通过，D_current/C同步；先前B下载中/CPU待过和N3未登录均历史。N1SSH73330/N2SSH31293/N3SSH97226，新三UUID及24h租期fresh门/2h保存继续，旧batch4所有端点永久禁连，新拒绝不得绕过。\n\n')
write(root/'automation_update_prepared.json',dict(id=auto['id'],mode='update',kind=auto['kind'],name=auto['name'],status=auto['status'],rrule=auto['rrule'],targetThreadId=auto['target_thread_id'],prompt=prefix+auto['prompt']))
print(json.dumps(dict(root=str(root),D_receipt=receipt,state_SHA=sha(current))))

import hashlib,json,pathlib,shutil,zipfile
P=pathlib.Path;r=P(__file__).resolve().parent;prior=r.parent/'candidate_posttrain_lowC_20261009T005229Z';out=r/'lease_window_actual_20261009T144258Z';out.mkdir();stamp='2026-10-09 14:42:58 UTC'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert shutil.disk_usage(out).free>45_000_000
names=['A100_audit_result.json','A100_audit_exit.json','A100_audit_capture.json','A100_audit_complete_original.zip','A100_audit_D_preservation_receipt.json','A100_inference_result.json','A100_inference_exit.json','A100_inference_capture.json','A100_inference_complete_original.zip','A100_fixed_official_VAL_TEST_prediction.npz','A100_inference_D_preservation_receipt.json','A100_prediction_publication.json','A100_prediction_publication_client_exit.json','A100_prediction_byte_audit.json','A100_prediction_byte_audit_exit.json','A100_prediction_byte_audit_capture.json','A100_prediction_byte_audit_complete_original.zip','A100_prediction_byte_audit_D_receipt.json','A100_lease_end_readonly_observation.json','B100_resume_transport_exit.json','B100_resume_transport_capture.json','B100_resume_transport_complete_original.zip','B100_resume_transport_stderr.log','B100_resume_transport_D_receipt.json','B100_lease_end_transport_observation.json','run_candidate_small_C_capture_v2.py']
for n in names:shutil.copy2(r/n,out/n)
for arc,cp,expected in [('A100_audit_complete_original.zip','A100_audit_capture.json',0),('A100_inference_complete_original.zip','A100_inference_capture.json',0),('A100_prediction_byte_audit_complete_original.zip','A100_prediction_byte_audit_capture.json',0),('B100_resume_transport_complete_original.zip','B100_resume_transport_capture.json',1)]:
 cap=read(out/cp);f=out/arc;assert sha(f)==cap['archive_SHA'] and f.stat().st_size==cap['archive_bytes'] and cap['natural_exit']==expected
 with zipfile.ZipFile(f) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));m=json.loads(z.read('member_SHA.json'))
  for n,h in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
inf=read(out/'A100_inference_result.json');pub=read(out/'A100_prediction_publication.json');byte=read(out/'A100_prediction_byte_audit.json');obs=read(out/'A100_lease_end_readonly_observation.json');assert inf['prediction_SHA']==byte['prediction_SHA']==sha(out/'A100_fixed_official_VAL_TEST_prediction.npz');assert pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED';assert not obs['score_once_token_exists'] and not obs['score_root_exists']
remaining=346223606-268435456
facts=dict(status='A_CPU_AND_OFFICIAL_PREDICTION_PEER_BYTE_PASS_B_TRANSPORT_FAIL_LEASE_SAVE_WINDOW',actualclock_snapshot_UTC=stamp,A_cpu4000_natural_exit0=True,A_prediction_natural_exit0=True,A_prediction_SHA=inf['prediction_SHA'],A_selected_epoch=73,A_official_prediction_Release_and_C_byte_audit_passed=True,A_score_stage_frozen_but_not_executed=True,A_score_original_observation=obs,B_resume_transport_natural_exit=1,B_failure_actual_UTC=read(out/'B100_resume_transport_exit.json')['actual_UTC'],B_failure_original_SHA=sha(out/'B100_resume_transport_complete_original.zip'),B_remaining_download_bytes=remaining,B_CPU4000_not_passed=True,new_OFFICIAL_VAL_TEST_five=None,formal_all_five_exceeded=False,conservative_not_platform_verified_lease_end_UTC='2026-10-09T15:00:00+00:00',execution_plus_2h_preservation_margin_unavailable=True,no_new_remote_stage_launched=True,no_lease_extension_or_release_or_shutdown=True,training_pair_complete_originals_still_public_Release_saved=True,extra_actual_replayed_updates={'A':3493,'B':3473},read_only_new_sessions={'A_SSH':79608,'C_SSH':50890,'C_SFTP':40971},closed_or_unknown_session_not_reused=[89661],not_all_research_complete=True)
write(out/'observed_facts.json',facts)
report='''A 的100轮/4000逻辑更新完整归档已通过异节点C审核：345 Adam状态均为4000，100轮严格earliest选模复核，第73轮CPUflow最大误差3.337860107421875e-06。审核原包SHA5161a0a4dd40ef5db0267973bb43d1d1ff90a8378b3641e545cc2b843055dfd2，完整D及Release保存。
A 同选定checkpoint官方VAL229/TEST685输入预测于UTC08:25:45自然0；NPZ SHA ece085400c2c80ac8b42f238b719a4c7ee8553ff4fe2fa7bb818d6e73f159d4f，原包SHA404c0991ecc2c89d98ffdcf0abac4ad4fdcdcff82f1e6eb1da9ef214408d7f6e，D+Release保存；C原数组字节审核832自然0，真实标签未参与，全部参数/RNG未变。
A 五项stage仅冻结并上传source，UTC14:42原远端只读确认score root及once token均不存在，没有实际评分结果。
B 新Range820于UTC08:34:50自然1，连续三次HTTP网络错误；最后分片收到268435456/346223606字节，还差77788150字节。失败原ZIP SHA301805083f0113ee1a9a008ea18b377b5e19c6d4a6e9b6b462b4be15f4156a06完整D SHA/CRC/unique/member通过，未完成CPU审核。
保守租期UTC15:00/北京时间23:00来自人类24h，非平台确认。当前不足执行加2h保存余量，未启动新远端阶段，不续租/释放/关机。所有已完成训练完整原ZIP及Release仍保留。新正式VAL/TEST五项、两臂比较、总体超过及全数据视频五折未完成；选模MSE不当MAE，TEST不择结构。额外实际重放A3493/B3473仍披露，非独立重复。
'''
(out/'research_current_report.md').write_text(report,encoding='utf8');shutil.copy2(__file__,out/P(__file__).name)
manifest={f.name:sha(f) for f in out.iterdir() if f.is_file()};write(out/'member_SHA.json',manifest);archive=out/'complete_local_evidence_snapshot.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n in list(manifest)+['member_SHA.json']:z.write(out/n,n)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
receipt=dict(status='CURRENT_ORIGINALS_D_SHA_CRC_UNIQUE_ALL_MEMBERS_SAVED',actualclock_before_snapshot_UTC=stamp,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,member_count=len(manifest)+1,not_remote_new_capture=True);write(out/'D_preservation_receipt.json',receipt)
sp=prior/'D_current_research_state.json';shutil.copy2(sp,out/'D_state_before_update.json');state=read(sp);state.update(status=facts['status'],updated_at_utc=stamp,latest_lease_window_actual=facts,latest_lease_window_D_receipt=receipt,remote_block_note='No current approval rejection. B actual bounded HTTP failure; conservative lease15:00 leaves insufficient execution+2h save margin. A CPU/prediction/peer-byte completed and retained, scoring not executed.',new_VAL_TEST_scores=None,formal_all_five_exceeded=False);write(sp,state)
cp=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs/自主优化实际接续.json')
assert shutil.disk_usage(cp.parent).free>201_000_000+sp.stat().st_size
shutil.copy2(cp,out/'C_state_before_update.json');shutil.copy2(sp,cp);assert sha(cp)==sha(sp)
with (cp.parent/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+stamp+' 最新短接续：A完整CPU749自然0、官方预测3629自然0、Release原预测NPZ ece085400…与原ZIP404c0991…通过、C独立原数组832自然0均已完整D过；官方评分尚未执行，14:42远端fresh UUID只读score root/once token均不存在。B接续820于08:34:50自然1，最后分片缺77788150字节，原失败301805083…全成员D过。保守15:00UTC/BJ23:00非平台确认，当前不足执行+2h保存余量，不启动远端新阶段，不续租/释放/关机。原训练完整ZIP/Release保留，暂无新正式五项。最新D事实 '+str(out/'observed_facts.json')+'；旧C89661 Unknown禁复用，新A79608/C50890+40971仅已fresh只读保存。\n')
print(json.dumps(receipt))

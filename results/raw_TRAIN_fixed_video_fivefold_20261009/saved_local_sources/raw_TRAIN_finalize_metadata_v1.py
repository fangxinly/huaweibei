import datetime,hashlib,json,pathlib,shutil,sys,os
P=pathlib.Path;r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def write(p,v):p.write_bytes(raw(v))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if sys.argv[2]=='--prepare':
 source=r/'saved_local_sources';source.mkdir()
 for n in ['raw_TRAIN_capture_v1.py','raw_TRAIN_save_and_publish_v1.py','raw_TRAIN_audit_report_v1.py','finalize_raw_TRAIN_actual_v1.py','prepare_raw_TRAIN_retry_v1.py','publish_raw_TRAIN_report_v1.py','raw_TRAIN_finalize_metadata_v1.py']:shutil.copyfile(P(__file__).with_name(n),source/n)
 report=P('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/TRAIN输入来源审核结论_20261010.md');assert sha(report)=='385aa951062f1af239bd5b0403bc4e0e96a6d1300405d52ac1df84ce2f9b6df2';shutil.copyfile(report,r/'independent_TRAIN_provenance_review_original.md')
 write(r/'independent_provenance_decision.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_SHA=sha(report),full_report_read=True,accept=['Old supervised frozen features are not strict pipeline OOF','Video grouping is not verified speaker identity','Old pooled split excluded','Row mapping does not erase supervision exposure'],main_source_chain_note='Main task has separately preserved original predecessor protocol/helper/author SHA and reconstructed row order; advisor audit scope does not negate those additional originals. Current raw fixed ridge is a distinct TRAIN-only protocol; old threefold mapping was not substituted for fixed fivefold manifest.',no_new_retraining_or_TEST_rescue=True))
 write(r/'analysis_batch_sent_receipt.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),threadId='01a10fcb-6663-70a2-9a76-60e5634d0c03',sent_once=True,original_tool_response={'threadId':'01a10fcb-6663-70a2-9a76-60e5634d0c03'},batch_archive_SHA='6c3799526decdedbe2ff328573e34d916593d2bcaf167597ddf87edef21c802f',analysis_only_no_credentials=True,full_reply_received=False,no_same_SHA_resend=True))
elif sys.argv[2]=='--sync':
 state=json.loads((dc/'D_current_research_state.json').read_bytes());pub=json.loads((dc/'GitHub_publication_receipt.json').read_bytes());state['github_source']=pub;state['latest_raw_TRAIN_analysis_sent']=json.loads((r/'analysis_batch_sent_receipt.json').read_bytes());state['latest_raw_TRAIN_probe_complete']['GitHub_publication_receipt']=pub;state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
  tmp=dest.with_name(dest.name+'.rawgithub.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
 adv=json.loads((c/'研究建议交流接续.json').read_bytes());adv['latest_raw_TRAIN_fixed_probe_analysis_sent']=state['latest_raw_TRAIN_analysis_sent'];write(c/'研究建议交流接续.json',adv)
 write(r/'final_GitHub_D_C_sync_receipt.json',dict(actual_UTC=state['updated_at_utc'],same_SHA=sha(dc/'D_current_research_state.json'),GitHub_commit=pub['commit']))
 assert sha(dc/'D_current_research_state.json')==sha(c/'自主优化实际接续.json');print(json.dumps(dict(same_SHA=sha(dc/'D_current_research_state.json'),GitHub_commit=pub['commit'])))
else:raise ValueError('Unknown mode')

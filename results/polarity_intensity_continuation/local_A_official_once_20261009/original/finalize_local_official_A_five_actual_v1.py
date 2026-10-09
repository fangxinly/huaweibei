import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
ev=Path(__file__).resolve().parent;prior=ev.parent/'candidate_posttrain_lowC_20261009T005229Z';root=ev/'A_official_local_five_saved_actual_20261009T145922Z';root.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert shutil.disk_usage('D:/').free>40*1024**2+40*1024**2;assert shutil.disk_usage('C:/').free>200*1024**2+40*1024**2
q=ev/'A_local_score_qualification_actual_20261009T145520Z';s=ev/'A_local_official_score_actual_20261009T145600Z';i=ev/'A_local_independent_metric_actual_20261009T145615Z';prep=ev/'local_official_score_preparation_20261009T145240Z';pub=ev/'A_local_official_publication_retry_actual_20261009T145820Z'
for r in (q,s,i):
    cap=read(r/'capture_receipt.json');assert cap['natural_exit']==0;assert sha(cap['archive'])==cap['archive_SHA']
    with zipfile.ZipFile(cap['archive']) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for row in json.loads(z.read('member_manifest.json'))['records']:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
result=read(s/'out/official_aligned_five_result.json');check=read(i/'out/independent_same_host_metric_check.json');publication=read(pub/'receipt.json')
assert publication['status']=='LOCAL_SMALL_ORIGINAL_RELEASE_ALL_REMOTE_DIGESTS_VERIFIED';assert read(pub/'natural_exit_tool_result.json')['exit_code']==0;assert check['max_error']<1e-12 and check['same_host_not_other_node']
files=[s/'out/official_aligned_five_result.json',i/'out/independent_same_host_metric_check.json',pub/'receipt.json',pub/'natural_exit_tool_result.json',prep/'local_official_protocol.json',Path(__file__)]
for r in (q,s,i):files +=[r/'capture_receipt.json',r/'natural_exit.json',r/'complete_actual_local_original.zip']
for p in prep.glob('*.py'):files.append(p)
files +=[ev/'A_local_official_publication_actual_20261009T145730Z/publication_receipt.json',ev/'A_local_official_publication_actual_20261009T145730Z/publication_plan.json',pub/'plan.json']
for p in files:
    dest=root/'original'/p.relative_to(ev);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(dest)==sha(p)
facts=dict(status='A_OFFICIAL_AUTHOR_FIVE_ONCE_LOCAL_COMPLETE_OTHER_NODE_METRICS_PENDING',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actualclock_before_finalize_UTC='2026-10-09 14:59:22 UTC',
    selected_epoch=73,selected_state_SHA=result['selected_state_SHA'],prediction_SHA=result['prediction_SHA'],dataset_SHA=result['official_dataset_SHA'],
    score=result,independent_same_host=check,other_node_metric_recalculation_pending=True,all_five_exceeded=False,
    TEST_point_improvements=['Acc2','F1'],TEST_point_worse=['Acc7','MAE','Corr'],VAL_point_improvements=['Acc2','F1','MAE','Corr'],VAL_point_worse=['Acc7'],
    Release=publication,B_CPU4000_not_passed=True,B_OFFICIAL_FIVE_not_scored=True,remote_original_A_score_stage_permanently_not_to_execute_duplicate=True,
    original_first_small_publication_URLError_preserved=True,no_new_remote_training_or_inference_or_score=True,not_all_research_complete=True)
write(root/'observed_facts.json',facts)
rows=['# A：官方一次五项，异节点指标复核待完成','', '选定第73轮；同一冻结预测、原官方数据SHA核验、作者指标源码原字节、NumPy1.26.4。新本地纯CPU评分协议先合成资格后真实一次评分，旧远端评分未执行且后续禁止重复。','', '|官方划分与模型|Acc7|Acc2|F1|MAE|Corr|','|---|---:|---:|---:|---:|---:|']
for role in ('VAL','TEST'):
    for label,key in [('A factorized_aux','new'),('CaReFlow','careflow')]:
        values=result['roles'][role][key];rows.append('|'+role+' '+label+'|'+'|'.join(f'{values[k]:.9f}' for k in ('Acc7','Acc2','F1','MAE','Corr'))+'|')
rows +=['','TEST仅Acc2/F1提高，Acc7/MAE/Corr落后，无五项整体超过。VAL四项提高但Acc7落后。','',f'独立标准库另进程同机复算max误差{check["max_error"]:.3g}；这是同机，不冒另一节点复核。异节点指标复核待完成。B传输失败原件保持，尚无完整CPU4000资格或五项。','', '额外重放A3493/B3473实际算力继续单列，TEST历史已看，不冒盲测、不按TEST选结构。原件D和Release保存；首小包上传URLError原件保留，有限小字节重试成功，无旧租期计算连接。']
(root/'research_current_report.md').write_text('\n'.join(rows)+'\n',encoding='utf8')
records=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob('*') if p.is_file()];write(root/'member_manifest.json',dict(records=records));archive=root/'complete_actual_local_current_original.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for row in records:z.write(root/row['path'],row['path'])
    z.write(root/'member_manifest.json','member_manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(records)+1
    for row in records:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
receipt=dict(status='A_LOCAL_OFFICIAL_FIVE_ALL_ORIGINALS_D_SAVED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,member_count=len(records),all_SHA_CRC_unique_passed=True,not_remote_capture=True,other_node_metric_recalculation_pending=True);write(root/'D_preservation_receipt.json',receipt)
statepath=prior/'D_current_research_state.json';state=read(statepath);shutil.copyfile(statepath,root/'D_state_before_update.json');state.update(status=facts['status'],new_VAL_TEST_scores=result['roles'],formal_all_five_exceeded=False,A_primary_official_score_once_complete=True,A_other_node_metric_check_pending=True,latest_local_A_five_actual=facts,latest_local_A_five_D_receipt=receipt,remote_block_note='A local author five scored once and D/Release retained; same-host independent calculation passed, other-node metric check pending. B transport failure remains. No new old-lease remote stage.');write(statepath,state)
c=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs/自主优化实际接续.json');shutil.copyfile(c,root/'C_state_before_update.json');shutil.copyfile(statepath,c);assert sha(c)==sha(statepath)
md=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs/研究接续状态.md')
with md.open('a',encoding='utf8') as f:f.write('\n2026-10-09 14:59:22 UTC 最新短接续：A本地纯CPU新评分协议先原数据/预测/source SHA及NumPy1.26.4/合成资格0，然后一次作者五项score0；独立标准库另进程同机0 max1.11e-16，不冒异节点指标复核完成。VAL .471615721/.888888889/.888099747/.593432132/.863577580；TEST .468613139/.880916031/.880732946/.638175758/.840997785。TEST仅Acc2/F1超过CaReFlow，三项落后，无整体五项超过。完整资格/计分/同机复算原ZIP已D全SHA/CRC/unique及Release远端digest，首URLError失败原件留存、有限小字节上传恢复成功。原A远端评分未执行且禁止以后重复，B完整CPU4000与五项仍未过，保守UTC15:00后禁本批连接无新有效期证据。最新原件 '+str(root)+'；TEST已看非盲测/不按TEST挑结构，重放算力A3493/B3473保持。\n')
print(json.dumps(dict(status=facts['status'],archive_SHA=receipt['archive_SHA'],root=str(root))))

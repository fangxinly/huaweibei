"""Seal local downloaded startup originals; never a remote COMPLETE capture."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile

stamp=sys.argv[1]
tag=stamp.replace('-','').replace(':','').replace(' ','T',1).replace(' UTC','Z')
home=Path.cwd()
base=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_formal_actual_dispatch_v3_20261007T015004Z')
dest=base.parent/('paired_formal_startup_local_seal_'+tag)
assert not dest.exists(), 'New immutable local seal only'
dest.mkdir()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
closure=home/'work/paired_formal_session_closure_20261007.json'
c=json.loads(closure.read_text(encoding='utf-8'))
assert len(c['results'])==6 and all(x['result']['status']=='fulfilled' and x['result']['value']['exit_code']==0 for x in c['results'])
shutil.copyfile(closure,base/'actual_interactive_session_closure.json')
report=home/'outputs/正式双方官方fullTRAIN100实际训练接续.json'
state=json.loads(report.read_text(encoding='utf-8-sig'))
state['actual_interactive_session_closure']={'clock_checked_utc':c['clock_checked_utc'],'source_D':str(base/'actual_interactive_session_closure.json'),'sha256':sha(closure),'all_six_explicit_exit_or_bye_code_zero':True,'all_old_ids_forbidden_to_reuse':True,'not_new_training_health_observation':True,'detached_training_not_stopped':True}
write(report,state)
line='\n交互连接已明确退出：A/B/C SSH45053/94972/62564与SFTP37470/61741/69706，原exit/bye工具结果均实际exit0；关闭观察clock '+c['clock_checked_utc']+'。所有旧ID禁复用；这是连接退出证据，不是新的实时训练健康检查，detached训练未停止。\n'
md=report.with_suffix('.md')
md.write_text(md.read_text(encoding='utf-8')+line,encoding='utf-8')
status=home/'outputs/研究接续状态.md'
status.write_text('本轮六个SSH/SFTP交互连接已原exit/bye实际0，全部旧ID禁复用；detached双方fullTRAIN100继续，01:55原观察不是关闭clock的新健康检查。\n\n'+status.read_text(encoding='utf-8'),encoding='utf-8')
files=[]
def copy_tree(src,label):
    for p in sorted(src.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        r=Path(label)/p.relative_to(src);q=dest/r;q.parent.mkdir(parents=True,exist_ok=True)
        before=sha(p);shutil.copyfile(p,q);assert sha(q)==before
        files.append({'member':r.as_posix(),'bytes':q.stat().st_size,'sha256':before,'actual_local_source':str(p)})
copy_tree(base,'downloaded_startup_originals_and_local_associations')
copy_tree(home/'work/paired_official_fulltrain_v2_20261007T011543Z','exact_frozen_source_bundle')
for name in ['研究接续状态.md','正式双方官方fullTRAIN100实际训练接续.md','正式双方官方fullTRAIN100实际训练接续.json','正式双方v2完整预检与保存实际接续.md','正式双方v2完整预检与保存实际接续.json','正式CaReFlow比较来源与预算本地准备接续.md','正式CaReFlow比较来源与预算本地准备接续.json']:
    p=home/'outputs'/name;q=dest/'continuation_snapshot'/name;q.parent.mkdir(exist_ok=True)
    h=sha(p);shutil.copyfile(p,q);assert sha(q)==h
    files.append({'member':q.relative_to(dest).as_posix(),'bytes':q.stat().st_size,'sha256':h,'actual_local_source':str(p)})
manifest={'status':'LOCAL_STARTUP_ORIGINALS_AND_FROZEN_SOURCE_SHA_SEAL_ONLY_NOT_REMOTE_COMPLETE_CAPTURE','actual_clock':stamp,'source_original_D':str(base),'members':files,'not_new_remote_capture':True,'not_completed_training':True,'no_final_weights_in_this_seal':True,'active_journal_prefix_bytes_not_downloaded':True,'original_observation_has_only_observed_prefix_hash_count_and_bytes':True,'no_model_forward_or_new_metrics':True,'old_full_precheck_weights_are_prior_physical_D_originals_not_new_downloads':True}
write(dest/'local_member_manifest.json',manifest)
archive=dest/'startup_local_snapshot.zip'
names=[x['member'] for x in files]+['local_member_manifest.json']
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for n in names:z.write(dest/n,n)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(names) and z.testzip() is None
    for x in files:assert hashlib.sha256(z.read(x['member'])).hexdigest()==x['sha256']
receipt={'status':'ACTUAL_LOCAL_STARTUP_D_SHA_ZIP_CRC_UNIQUE_SEAL_PASSED_NOT_REMOTE_COMPLETE_CAPTURE','actual_clock':stamp,'D_root':str(dest),'snapshot_bytes':archive.stat().st_size,'snapshot_sha256':sha(archive),'manifest_sha256':sha(dest/'local_member_manifest.json'),'unique_members':len(names),'ZIP_CRC_all_passed':True,'all_member_SHA_and_source_copy_passed':True,'formal100_complete':False,'new_remote_CAPTURE_COMPLETE_claimed':False,'new_CPU_forward_or_final_weight_download_claimed':False,'session_closure_sha256':sha(closure)}
write(dest/'actual_local_startup_seal_receipt.json',receipt)
state['actual_local_startup_D_seal_receipt']={'file':str(dest/'actual_local_startup_seal_receipt.json'),'sha256':sha(dest/'actual_local_startup_seal_receipt.json'),'not_remote_COMPLETE_capture':True}
write(report,state)
md.write_text(md.read_text(encoding='utf-8')+'\n本地启动证据封存D '+str(dest)+'；snapshot SHA '+receipt['snapshot_sha256']+'，'+str(len(names))+'唯一成员/全SHA/ZIPCRC通过。仅保存已下载启动原件/源/接续快照，非remote COMPLETE capture、非final weights、非新CPU前向或成绩；active journal只原观察SHA/count/bytes，前缀字节尚未下载。\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))

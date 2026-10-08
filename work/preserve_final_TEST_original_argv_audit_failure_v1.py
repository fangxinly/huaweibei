"""Keep original successful predictions/captures and failed D audits; no model or labels."""
import datetime,hashlib,json,shutil,zipfile
from pathlib import Path

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    root=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_final_TEST_prediction_actual_20261007T043535Z')
    frozen=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_final_TEST_pipeline_frozen_20261007T043535Z')
    nodes={}
    expected={'A':'6d42e21f76d2cd4d37f53556802fb794e0ac55d39ad143c3d4386238a43e14e1',
              'C':'aec92717d6c0caa561b5702ee01513e0495f18ae5ceca17b4a28be7755f9d0e2'}
    for node in ('A','C'):
        d=root/node;r=read(d/'capture_receipt.json');x=read(d/'capture_actual_natural_exit.json')
        assert sha(d/'capture_receipt.json')==expected[node]==x['capture_receipt_sha256']
        assert r['snapshot_sha256']==sha(d/'snapshot.zip') and r['manifest_sha256']==sha(d/'original_member_manifest.json')
        assert r['native_sha256']==sha(d/'actual_capture_native_preflight.json')
        assert x['stdout_sha256']==sha(d/'capture.stdout.log') and x['stderr_sha256']==sha(d/'capture.stderr.log')
        assert x['natural_exit_code']==0 and r['original_child_exit_code']==0
        manifest=read(d/'original_member_manifest.json')['members']
        with zipfile.ZipFile(d/'snapshot.zip') as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)==66
            assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] and len(z.read(n))==v['bytes'] for n,v in manifest.items())
            launch=json.loads(z.read('actual_child_launch.json'));stage=json.loads(z.read('out/actual_stage_receipt.json'))
            exitdata=json.loads(z.read('natural_exit.json'));plan=json.loads(z.read('source/final_pair_execution_plan.json'))
            assert exitdata['natural_exit'] is True and exitdata['exit_code']==0 and exitdata['child_pid']==launch['child_pid']==stage['pid']
            assert hashlib.sha256(z.read('out/actual_stage_receipt.json')).hexdigest()==exitdata['stage_receipt_sha256']
            assert all(hashlib.sha256(z.read('source/'+n)).hexdigest()==h for n,h in plan['source_sha256'].items())
            assert hashlib.sha256(z.read('source/final_pair_execution_plan.json')).hexdigest()=='f41d05150d5ff1522c45508869981ea9e878aa75327e79f8a6895a0323ef62b6'
            assert x['capture_child_fullargv'][1:]==r['capture_fullargv'] and launch['fullargv'][1:]==stage['fullargv']
            assert stage['labels_read'] is False and stage['dummy0vs7_exact_equal'] is True
            assert stage['prediction_sha256']==stage['dummy_prediction_sha256']==hashlib.sha256(z.read('out/prediction.npz')).hexdigest()
            nodes[node]={'method':stage['method'],'original_root':stage['root'],'original_child_pid':stage['pid'],
                         'original_child_natural_exit_UTC':exitdata['actual_utc'],'original_capture_actual_UTC':r['actual_utc'],
                         'original_capture_natural_exit_UTC':x['actual_utc'],'prediction_sha256':stage['prediction_sha256'],
                         'state_sha256':stage['state_sha256'],'selected_checkpoint_sha256':stage['checkpoint_sha256'],
                         'original_stage_receipt_sha256':exitdata['stage_receipt_sha256'],
                         'capture_receipt_sha256':sha(d/'capture_receipt.json'),'snapshot_sha256':sha(d/'snapshot.zip'),
                         'all66_original_source_member_bytes_SHA_CRC_unique_passed':True,
                         'dummy0vs7_state_allRNG_unchanged_original_receipt':True,
                         'cumulative_peak_no_reset':stage['cumulative_peak_no_reset'],
                         'labels_read':False,'D_original_process_association_auditor_natural_exit':1,
                         'D_original_process_association_full_gate_passed':False,'other_CPU_audit_executed':False}
    closed=read(root/'operator/actual_closed_session_full_tool_records.json')
    assert len(closed)==6 and all(x['tool']['status']=='fulfilled' and x['tool']['value']['exit_code']==0 for x in closed)
    record={'status':'ACTUAL_BOTH_FIXED_TEST685_PREDICTIONS_NATURAL0_ORIGINAL_CAPSULES_D_SAVED_PROCESS_ARGV_D_AUDITOR_FAILED_NO_CPU_OR_SCORE',
            'actual_local_record_UTC':now(),'D':str(root),'nodes':nodes,
            'execution_plan_sha256':'f41d05150d5ff1522c45508869981ea9e878aa75327e79f8a6895a0323ef62b6',
            'frozen_source_D':str(frozen),'frozen_source_seal_receipt_sha256':sha(frozen/'complete_local_preparation_seal_receipt.json'),
            'failure':'D auditor compares process argv including interpreter with recorded sys.argv excluding interpreter; both capture and model-child associations reject. Original files remain unchanged.',
            'failure_exact_process_PID_or_UTC_not_available_from_shell_tool':True,
            'failure_original_full_tool_records':[str(root/'operator'/('original_D_audit_failure_tool_'+str(i)+'.json')) for i in range(2)],
            'readonly_original_diagnosis_sha256':sha(root/'operator/readonly_argv_representation_diagnosis.json'),
            'no_original_file_or_frozen_source_changed':True,'no_prediction_rerun_or_target_read':True,
            'no_model_forward_this_local_seal':True,'other_CPU_and_full_prediction_joint_not_passed':True,
            'all_six_sessions_explicit_exit_bye0_closed':[x['session_id'] for x in closed],
            'next':'Separate exact source audit correction of interpreter versus sys.argv representation, preserve rejected audit and all other original checks; no prediction/201/training rerun, no score before both completed D/other CPU/capture joints. Frozen failure policy stopped further stages this batch.',
            'overall_five_goal_final_score_and_lease_strong_saves_complete':False,
            'pending_lease_strong_actual_saves_UTC':['09:30','11:30','13:00'],
            'directory_suffix_is_frozen_protocol_batch_identity_not_local_capture_time':True}
    write(root/'actual_prediction_and_D_audit_failure_continuation.json',record)
    shutil.copyfile(Path(__file__),root/'operator/original_failure_preservation_source.py')
    files=sorted(p for p in root.rglob('*') if p.is_file())
    manifest={str(p.relative_to(root)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    write(root/'complete_original_batch_member_manifest.json',manifest);files.append(root/'complete_original_batch_member_manifest.json')
    with zipfile.ZipFile(root/'complete_original_prediction_and_audit_failure.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(root)).replace('\\','/'))
    with zipfile.ZipFile(root/'complete_original_prediction_and_audit_failure.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)
        assert all(hashlib.sha256(z.read(n)).hexdigest()==sha(root/n) for n in z.namelist())
    seal={'status':'ACTUAL_ORIGINAL_PREDICTIONS_AND_FAILED_AUDIT_BATCH_LOCAL_D_COMPLETE_SHA_CRC_UNIQUE_MEMBER_SHA_SEALED',
          'actual_UTC':now(),'zip_sha256':sha(root/'complete_original_prediction_and_audit_failure.zip'),
          'members':len(files),'continuation_sha256':sha(root/'actual_prediction_and_D_audit_failure_continuation.json'),
          'D_free_bytes':shutil.disk_usage(root).free,'not_remote_capture_or_full_prediction_joint_success':True}
    write(root/'complete_original_prediction_and_audit_failure_seal_receipt.json',seal)
    output=dict(record,local_seal_receipt_sha256=sha(root/'complete_original_prediction_and_audit_failure_seal_receipt.json'))
    Path('outputs/正式双方TEST685预测实际完成与保存审核失败接续.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md='固定双方TEST685零真标签预测已真实完成：F child3848 UTC04:39:06.738646、CaReFlow child1465 UTC04:39:07.869652，自然0；dummy0/7原预测SHA各自完全相同，原receipt记全state/stat/RNG不变。两66成员完整capture自然0且完整字节/原source逐项SHA/ZIPCRC/唯一已D保存。\n\nD过程关联审核均自然1：源码错误地直接比较带Python解释器的进程argv与不带解释器的sys.argv。只读原件诊断确认两处差别均仅解释器前缀；这不是预测GPU失败或审批拒绝。失败原tool完整记录和所有原件已D封存，旧冻结源未改，不重跑预测。尚未完成B原CPU、完整双预测联合门或一次真标签评分。六会话明确exit/bye0均关闭禁复用。\n\n下一步只能另冻限定过程参数表示修正审核，保留原拒绝及全部其他检查；不能据诊断冒审核通过，不能读真标签救分。09:30/11:30/13:00租期动态保存与整体五项目标仍未完成。\n\nD：'+str(root)+'\n完整本地seal SHA：'+output['local_seal_receipt_sha256']+'\n'
    Path('outputs/正式双方TEST685预测实际完成与保存审核失败接续.md').write_text(md,encoding='utf-8')
    state=Path('outputs/研究接续状态.md')
    prefix='最新原结果与保存审核新失败 '+seal['actual_UTC']+'：F3848/C1465 TEST685固定零标签预测自然0，双dummy预测字节SHA相同，两66原capture自然0→完整D字节/源SHA/CRC/唯一；D过程关联审核各自然1，原因full进程argv包含解释器而原sys.argv不含，原tool/readonly诊断/全部原件完整D封存。不冒完整联合通过、B CPU或真标签评分；不重跑预测/旧实验/改旧冻结源。先读《正式双方TEST685预测实际完成与保存审核失败接续.md/json》。6会话70598/36873/31965/41423/77453/73800全明确exit/bye0禁复用；后续另冻限定审核表示修正，双方联合门前禁标签。租期强化保存和整体五项目标未完成。\n\n'
    state.write_text(prefix+state.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'status':record['status'],'D':str(root),'seal':seal,'seal_receipt_sha256':output['local_seal_receipt_sha256']},ensure_ascii=False))

if __name__=='__main__':main()

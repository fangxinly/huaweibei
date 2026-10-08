"""Join a new genuine fresh-instance capture with the already physically saved full-training joint."""
import argparse
from pathlib import Path
import sys
from paired_fulltrain_evidence_candidate_v1 import require,now,sha,read,write,plan_gate,capture_association,stage_association,TRAIN_JOINT

def main():
    p=argparse.ArgumentParser()
    for n in ('D-root','fresh-capture','training-joint','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--training-joint-sha',required=True);p.add_argument('--plan-sha',required=True)
    a=p.parse_args();root=a.D_root.resolve();base=a.fresh_capture.resolve()
    require(root.drive.upper()=='D:' and base.is_relative_to(root) and a.out.resolve().is_relative_to(root) and not a.out.exists(),'Fresh result must be inside actual D preservation root')
    require(sha(a.training_joint)==a.training_joint_sha,'Exact original whole training joint SHA')
    old=read(a.training_joint);require(old['status']==TRAIN_JOINT and old['whole_checkpoint_D_SHA_CRC_passed'] is True,'Original whole training joint absent')
    c,m=capture_association(base);plan=plan_gate(base/'source',a.plan_sha)
    r,e=stage_association(base/'run',a.plan_sha,old['method'])
    require(r['status']=='ACTUAL_PAIRED_FRESH_PUBLIC_INSTANCE_WHOLE_SELECTED_REPLAY_COMPLETE_NO_LABELS_OR_FINAL_TEST' and
            r['original_training_joint_sha256']==a.training_joint_sha and r['original_stage_receipt_sha256']==old['original_stage_receipt_sha256'] and
            r['original_prediction_replay_error']==0 and r['dummy0vs7_error']==0 and r['state_and_allRNG_unchanged'] is True and
            r['new_label_or_final_TEST_access'] is False and not any(x.get('labels_read') for x in r['guard_journal']), 'Fresh actual replay/source/label gate mismatch')
    checkpoint=old['original_complete_checkpoints_D']['selected_best_full']
    ref=r['external_checkpoint_reference'];capture_ref=m['large_references_not_downloads']['reference/selected_best_full.pt']
    require(ref['sha256']==capture_ref['sha256']==checkpoint['sha256'] and ref['bytes']==capture_ref['bytes']==checkpoint['bytes'], 'Fresh entire selected/reference/D model association')
    wrapper=read(base/'run/wrapper_actual_start.json')
    require(wrapper['child_source_sha256']==plan['source_sha256']['paired_fulltrain_fresh_selected_replay_candidate_v1.py'], 'Actual fresh child frozen source mismatch')
    result={'status':'ACTUAL_PAIRED_FULLTRAIN_D_ORIGINAL_OTHER_CPU_AND_FRESH_PUBLIC_SELECTED_REPLAY_JOINT_PASSED_NO_FINAL_TEST',
            'actual_local_joint_utc':now(),'argv':sys.argv,'method':old['method'],'plan_sha256':a.plan_sha,
            'whole_training_joint_sha256':a.training_joint_sha,'original_stage_receipt_sha256':old['original_stage_receipt_sha256'],
            'original_other_CPU_receipt_sha256':old['original_other_CPU_receipt_sha256'],
            'fresh_receipt_sha256':sha(base/'run/out/actual_stage_receipt.json'),'fresh_exit_sha256':sha(base/'run/natural_exit.json'),
            'fresh_capture_receipt_sha256':sha(base/'capture_receipt.json'),'fresh_capture_exit_sha256':sha(base/'capture_actual_exit.json'),
            'fresh_complete_ZIP_sha256':c['snapshot_sha256'],'fresh_actual_capture_utc':c['actual_utc'],
            'selected_whole_file_SHA':checkpoint['sha256'],'all_original_model_and_array_gates_passed':True,
            'CPU_model_forward':False,'fresh_GPU_model_forward':True,'large_references_not_new_downloads':True,
            'final_TEST_executed':False,'new_five_metric_superiority_or_overall_goal_complete':False}
    write(a.out,result);print(__import__('json').dumps(result),flush=True)

if __name__=='__main__':main()

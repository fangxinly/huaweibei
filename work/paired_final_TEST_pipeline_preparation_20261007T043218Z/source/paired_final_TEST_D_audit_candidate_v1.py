"""Validate complete original remote capsule before extracting; no assets/model/labels."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import zipfile
from paired_final_TEST_pipeline_candidate_v1 import plan_gate,load,require,sha,write,utc,original_prediction_gate,CPU_STATUS

def audit_capsule(dest, expected_plan, expected_node):
    dest=Path(dest);r=load(dest/'capture_receipt.json');x=load(dest/'capture_actual_natural_exit.json')
    require(r['status']=='CAPTURE_COMPLETE_FINAL_TEST_ORIGINAL_SUCCESS' and
            r['plan_sha256']==expected_plan and r['node']==expected_node and r['original_child_exit_code']==0,
            'Original successful complete capsule required')
    require(x['natural_exit_code']==0 and x['capture_child_pid']==r['capture_child_pid'] and
            x['capture_child_fullargv']==r['capture_fullargv'] and
            x['capture_receipt_sha256']==sha(dest/'capture_receipt.json') and
            x['stdout_sha256']==sha(dest/'capture.stdout.log') and x['stderr_sha256']==sha(dest/'capture.stderr.log'),
            'Capture actual natural exit/source/full logs association')
    require(sha(dest/'snapshot.zip')==r['snapshot_sha256'] and
            (dest/'snapshot.zip').stat().st_size==r['snapshot_bytes'] and
            sha(dest/'original_member_manifest.json')==r['manifest_sha256'] and
            sha(dest/'actual_capture_native_preflight.json')==r['native_sha256'],'Original complete capsule byte linkage')
    manifest=load(dest/'original_member_manifest.json')
    require(manifest['original_root']==r['original_root'],'Original manifest root')
    members=manifest['members']
    with zipfile.ZipFile(dest/'snapshot.zip') as z:
        require(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==r['members'] and
                set(z.namelist())==set(members),'All original ZIP CRC/unique/coverage')
        for name,item in members.items():
            parts=PurePosixPath(name).parts
            require(not PurePosixPath(name).is_absolute() and '..' not in parts and '\\' not in name and
                    ':' not in name, 'Unsafe original member path')
            b=z.read(name);require(len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'], 'Original member SHA')
        root=dest/'run';require(not root.exists(),'Do not overwrite old D originals');root.mkdir()
        z.extractall(root)
    plan=plan_gate(root/'source',expected_plan)
    launch,exitdata=load(root/'actual_child_launch.json'),load(root/'natural_exit.json')
    require(exitdata['child_pid']==r['original_child_pid']==launch['child_pid'] and
            exitdata['fullargv']==r['original_child_fullargv']==launch['fullargv'] and
            exitdata['exit_code']==0 and exitdata['natural_exit'] is True and
            sha(root/'natural_exit.json')==r['natural_exit_sha256'] and
            exitdata['original_launch_sha256']==sha(root/'actual_child_launch.json') and
            sha(root/'child.stdout.log')==exitdata['stdout_sha256'] and
            sha(root/'child.stderr.log')==exitdata['stderr_sha256'],'D original child natural0/source/full logs')
    stage=load(root/'out/actual_stage_receipt.json')
    require(sha(root/'out/actual_stage_receipt.json')==exitdata['stage_receipt_sha256'] and
            stage['pid']==launch['child_pid'] and stage['fullargv']==launch['fullargv'] and
            stage['root']==r['original_root'] and stage['plan_sha256']==expected_plan,'D stage receipt association')
    if launch['stage']=='predict':
        original_prediction_gate(root,plan,expected_plan,launch['method'],sha(root/'source'/plan['identity_file']))
    elif launch['stage']=='cpu':
        require(stage['status']==CPU_STATUS and stage['node']=='B' and stage['labels_read'] is False,
                'Original independent CPU audit required')
        parent=original_prediction_gate(root/'original',plan,expected_plan,launch['method'],sha(root/'source'/plan['identity_file']))
        require(stage['original_receipt_sha256']==sha(root/'original/out/actual_stage_receipt.json') and
                stage['original_natural_exit_sha256']==sha(root/'original/natural_exit.json') and
                stage['prediction_sha256']==parent['prediction_sha256'],'CPU original arrays/receipts exact linkage')
    else:
        require(stage['status']=='ACTUAL_FIXED_AUTHOR_CACHE_TEST685_ONCE_JOINT_FIVE_SCORE_COMPLETE' and
                stage['labels_read'] is True,'Original once score required')
    native=load(dest/'actual_capture_native_preflight.json')
    require(native['actual_asset_sha256']==plan['asset_sha256'] and
            native['native_raw']['UUID'].strip()==plan['allowed_node_UUID'][expected_node] and
            not native['native_raw']['compute'].strip(),'Actual capsule fresh native assets/UUID')
    result={'status':'ACTUAL_FINAL_TEST_COMPLETE_CAPSULE_D_SOURCE_ARGV_SHA_CRC_PASSED',
            'actual_utc':utc(),'stage':launch['stage'],'method':launch['method'],'node':expected_node,
            'plan_sha256':expected_plan,'original_root':r['original_root'],
            'stage_receipt_sha256':sha(root/'out/actual_stage_receipt.json'),'snapshot_sha256':r['snapshot_sha256'],
            'capture_receipt_sha256':sha(dest/'capture_receipt.json'),'original_child_pid':r['original_child_pid'],
            'original_child_natural_exit_code':0,'complete_members':len(members),'labels_read':stage['labels_read']}
    write(dest/'actual_D_saved_capsule_audit.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',required=True)
    a=p.parse_args();print(audit_capsule(a.dest,a.plan_sha,a.node))

if __name__=='__main__':main()

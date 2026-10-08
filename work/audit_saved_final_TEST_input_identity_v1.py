"""Actual local receipt/whole capsule audit; reads input identity JSON only."""
import argparse,datetime,hashlib,json,shutil,zipfile
from pathlib import Path,PurePosixPath
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(v,msg):
    if not v:raise ValueError(msg)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--saved',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--node',choices=('A','B'),required=True)
    ap.add_argument('--original-A',type=Path);a=ap.parse_args()
    planfile=a.bundle/'input_identity_plan.json';ph=sha(planfile);plan=read(planfile)
    require(ph=='d341548ebaff66125c450b42f38d1c19bacba5e004450ef8bc41dfe68e4d8f2a','Exact frozen input-only plan')
    p=a.saved;c=read(p/'capture_receipt.json');e=read(p/'capture_actual_natural_exit.json')
    require(e['original_capture_receipt_sha256']==sha(p/'capture_receipt.json') and e['natural_exit_code']==0 and
            c['status']=='CAPTURE_COMPLETE_ORIGINAL_INPUT_ONLY_SUCCESS' and c['original_child_exit_code']==0 and
            c['node']==a.node and c['plan_sha256']==ph,'Original capture COMPLETE/natural0/plan linkage')
    require(sha(p/'capture.stdout.log')==e['stdout_sha256'] and sha(p/'capture.stderr.log')==e['stderr_sha256'],
            'Original actual capture stdout/stderr')
    require(c['capture_child_pid']==e['capture_child_pid'] and c['capture_fullargv']==e['capture_child_fullargv'][1:],
            'Original capture child actualargv/PID')
    for name,key in (('snapshot.zip','snapshot_sha256'),('original_member_manifest.json','member_manifest_sha256'),
                     ('actual_capture_native_preflight.json','direct_native_capture_sha256')):
        require(sha(p/name)==c[key],'Receipt-first complete saved artifact SHA: '+name)
    m=read(p/'original_member_manifest.json');require(m['root']==c['original_root'],'Original root mismatch')
    native=read(p/'actual_capture_native_preflight.json')
    require(native['native_raw']['gpu_UUID'].strip()==plan['allowed_node_UUID'][a.node] and
            not native['native_raw']['compute'].strip() and native['original_assets_checked_sha256']==plan['asset_sha256'],
            'Capture original assigned physical identity/assets')
    dst=p/'run';require(not dst.exists(),'Do not overwrite extracted originals')
    with zipfile.ZipFile(p/'snapshot.zip') as z:
        require(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==c['members'] and
                set(z.namelist())==set(m['members']),'Full CRC/unique/all original members')
        for name,spec in m['members'].items():
            parts=PurePosixPath(name)
            require(not parts.is_absolute() and '..' not in parts.parts and '\\' not in name,'Safe original member path')
            data=z.read(name)
            require(len(data)==spec['bytes'] and hashlib.sha256(data).hexdigest()==spec['sha256'],'Original member bytes/SHA')
        z.extractall(dst)
    for name,spec in m['members'].items():require(sha(dst/name)==spec['sha256'],'Extracted original SHA')
    require(sha(dst/'source/input_identity_plan.json')==ph,'Original root frozen plan')
    for n,h in plan['source_sha256'].items():require(sha(dst/'source'/n)==sha(a.bundle/n)==h,'All original exact source/parents')
    launch=read(dst/'actual_child_launch.json');exitdata=read(dst/'natural_exit.json');r=read(dst/'out/actual_stage_receipt.json')
    require(sha(dst/'natural_exit.json')==c['original_natural_exit_sha256'] and exitdata['exit_code']==0 and
            exitdata['natural_exit'] is True and exitdata['child_pid']==launch['child_pid']==r['pid']==c['original_child_pid'] and
            exitdata['fullargv']==launch['fullargv']==c['original_child_fullargv'] and r['fullargv']==launch['fullargv'][1:],
            'Original successful identity child source/fullargv/PID/natural exit')
    require(sha(dst/'out/actual_stage_receipt.json')==exitdata['original_stage_receipt_sha256'] and
            r['plan_sha256']==ph and r['source_sha256']==plan['source_sha256'] and r['node']==a.node and
            r['status']=='ACTUAL_OFFICIAL_TEST_INPUT_ONLY_IDENTITY_COMPLETE' and r['all_role_labels_read'] is False and
            r['TEST_prediction_or_model_forward_or_scoring_executed'] is False,'Original receipt exact input-only role')
    require(sha(dst/'out/actual_native_preflight.json')==r['native_preflight_sha256'],'Original pre-access native gate')
    initial=read(dst/'out/actual_native_preflight.json')
    require(initial['native_raw']['gpu_UUID'].strip()==plan['allowed_node_UUID'][a.node] and
            not initial['native_raw']['compute'].strip() and initial['original_assets_checked_sha256']==plan['asset_sha256'],
            'Before-input physical/source/asset gate')
    ids={role:r[role+'_ids'] for role in ('train','dev','test')}
    require(len(ids['train'])==1281 and len(ids['dev'])==229 and len(ids['test'])==r['test_rows'] and
            all(len(v)==len(set(v)) for v in ids.values()),'Original complete IDs/counts')
    for x,y in (('train','dev'),('train','test'),('dev','test')):require(not set(ids[x])&set(ids[y]),'Original role overlap')
    linked_A=None
    if a.node=='B':
        require(a.original_A is not None,'B must link original saved A')
        original=read(a.original_A);linked_A=sha(a.original_A)
        require(r['original_A_identity_sha256']==sha(dst/'original_A_input_identity.json')==linked_A and
                r['independent_raw_input_ID_order_equality_passed'] is True,'B exact original A bytes linkage')
        for key in ('train_ids','dev_ids','test_ids','test_rows','test_raw_input_sha256','test_original_input_layouts',
                    'prior_official_train_dev_ID_sha256','all_role_labels_read'):
            require(original[key]==r[key],'B original input/ID equality: '+key)
    result={'status':'ACTUAL_TEST_INPUT_ONLY_SOURCE_ROOT_ARGV_CAPTURE_D_SHA_CRC_PASSED',
            'actual_local_receipt_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'node':a.node,'source_plan_sha256':ph,'original_root':c['original_root'],
            'original_child_pid':r['pid'],'original_child_natural_exit_UTC':exitdata['actual_utc'],
            'original_capture_UTC':c['actual_utc'],'original_capture_natural_exit_UTC':e['actual_utc'],
            'saved_identity_path':str(dst/'out/actual_stage_receipt.json'),
            'saved_identity_sha256':sha(dst/'out/actual_stage_receipt.json'),
            'snapshot_sha256':c['snapshot_sha256'],'all_source_fullargv_natural_exit_members_CRC_SHA_passed':True,
            'snapshot_members':c['members'],'test_rows':r['test_rows'],'test_raw_input_sha256':r['test_raw_input_sha256'],
            'other_CPU_original_A_sha256':linked_A,'TEST_labels_read_or_model_forward':False,
            'canonical_paper_official_ID_coverage_not_proven_by_cache_role_only':True,
            'large_model_bytes_not_retransferred':True,'D_free_bytes':shutil.disk_usage('D:/').free}
    output=p/'actual_D_saved_input_identity_audit.json';require(not output.exists(),'Do not overwrite actual audit')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(result,audit_sha256=sha(output)),ensure_ascii=False))

if __name__=='__main__':main()

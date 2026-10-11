"""Independent transport and actual collection-subroot coverage audit."""
from pathlib import Path
import argparse, datetime, hashlib, json, zipfile
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
p.add_argument('--collection',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True)
p.add_argument('--prior-map',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
prior_map=read(a.prior_map);reports={}
dynamic={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'}
for fold,node in enumerate(['a','b','c']):
    d=a.directory/node;original=a.collection/node;r=read(d/'receipt.json')
    assert (d/'exit_code.txt').read_text().strip()=='0'
    exit_proof=read(d/'actual_capture_exit.json');assert exit_proof['exit_code']==0
    raw=(d/'snapshot.zip').read_bytes();assert len(raw)==r['bytes'] and sha(raw)==r['sha256']
    with zipfile.ZipFile(d/'snapshot.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        m=json.loads(z.read('member_manifest.json'))
        assert set(z.namelist())==set(m)|{'member_manifest.json'}
        for name,meta in m.items():
            data=z.read(name);assert len(data)==meta['bytes'] and sha(data)==meta['sha256']
        assert sha(z.read('source/capture_soft_vector_v19.py'))==exit_proof['capture_source_sha256']=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
        plan=read(original/'same_teacher_collection_plan_v2.json');selected=plan['folds'][fold]
        inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0].strip()==selected['expected_uuid'] and not inv['compute']
        prefix='group_teacher/'+plan['collection_subdirectory']+'/'
        required=set(plan['new_source_sha256'])|{'same_teacher_collection_plan_v2.json','run_same_teacher_collection_v2.py','audit_same_teacher_collection_v2.py','calibration_video_role_plan_v1.json','deployment_verified.json','precheck_exit.json','precheck_process.log','precheck/inventory.json','precheck/receipt.json','precheck/independent_precheck_audit.json'}
        required.update(selected['role_files'])
        if a.phase=='execute':
            required.update({'execute_exit.json','execute_process.log','execute/inventory.json','execute/receipt.json','execute/fit_scalar_inputs.npz','execute/inner_scalar_inputs.npz','execution_authorization.json'})
        for name in required:
            assert z.read(prefix+name)==(original/name).read_bytes(),name
        assert read(original/'precheck_exit.json')['exit_code']==0
        if a.phase=='execute':assert read(original/'execute_exit.json')['exit_code']==0
        with zipfile.ZipFile(prior_map[node]) as old:
            old_m=json.loads(old.read('member_manifest.json'))
            old_l=json.loads(old.read('large_file_manifest.json'))
        unchanged=0
        for name,meta in old_m.items():
            if name in dynamic:continue
            assert m[name]==meta,'Old source/evidence changed: '+name
            unchanged+=1
        large=json.loads(z.read('large_file_manifest.json'))
        for name,meta in old_l.items():
            assert large[name]['sha256']==meta['sha256'] and large[name]['bytes']==meta['bytes'],name
        reports[node]={'capture_utc':r['utc'],'zip_sha256':r['sha256'],'members':len(m),'new_required_original_members':len(required),
                       'old_frozen_members_unchanged':unchanged,'old_large_sha_references_unchanged':len(old_l),
                       'complete_phase_exit_code':0,'fresh_weight_sha_is_new_download':False}
result={'status':'ACTUAL_COLLECTION_SUBROOT_SOURCE_PROCESS_RECEIPTS_ARRAYS_AND_PRIOR_FROZEN_EVIDENCE_CAPTURE_VERIFIED',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':a.phase,'nodes':reports,'source_sha256':sha(Path(__file__).read_bytes())}
assert not a.out.exists();a.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(result['status'])

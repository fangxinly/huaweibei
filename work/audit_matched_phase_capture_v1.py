from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_text())
rec=read(a.snapshot/'receipt.json');ex=read(a.snapshot/'actual_capture_exit.json');assert ex['exit_code']==0 and (a.snapshot/'exit_code.txt').read_text().strip()=='0'
b=(a.snapshot/'snapshot.zip').read_bytes();assert len(b)==rec['bytes'] and sha(b)==rec['sha256']
plan=read(a.root/'plan.json');prefix='group_teacher/'+Path(plan['new_root']).name+'/'
base=['plan.json','cal_parameters.json','roles.npz','mu_input.npz','diagnose_matched_message_pools_v1.py','audit_matched_message_pools_v1.py','run_matched_message_pools_phase_v2.py','run_same_teacher_capture_v1.py','evaluation_plan.json','evaluate_matched_message_pools_v1.py','audit_matched_pool_eval_v1.py']
original_phase_files=['receipt.json','inventory.json','prediction_freeze.json','predictions_frozen.npz','independent_audit.json']
required=base+['precheck_launch.json','precheck_exit.json','precheck.log','precheck_wrapper.log']+['precheck/'+name for name in original_phase_files]
if a.phase=='execute':required+=['execution_authorization.json','execute_launch.json','execute_exit.json','execute.log','execute_wrapper.log']+['execute/'+name for name in original_phase_files]
with zipfile.ZipFile(a.snapshot/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    m=json.loads(z.read('member_manifest.json'));large=json.loads(z.read('large_file_manifest.json'))
    assert set(z.namelist())==set(m)|{'member_manifest.json'}
    for name,item in m.items():
        b=z.read(name);assert len(b)==item['bytes'] and sha(b)==item['sha256'],name
    assert sha(z.read('source/capture_soft_vector_v19.py'))==ex['capture_source_sha256']=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
    inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0].strip()==plan['expected_uuid'] and not inv['compute']
    newlarge=[]
    for name in required:
        bb=(a.root/name).read_bytes();key=prefix+name
        if key in m:assert z.read(key)==bb,key
        else:assert large[key]['sha256']==sha(bb) and large[key]['bytes']==len(bb),key;newlarge.append(key)
    with zipfile.ZipFile(a.prior) as old:om=json.loads(old.read('member_manifest.json'));ol=json.loads(old.read('large_file_manifest.json'))
    oldexit=read(a.prior.parent/'actual_capture_exit.json');assert oldexit['exit_code']==0
    stamp=oldexit['argv'][-1];closure=[name for name in om if name.endswith('/capture_'+stamp+'.log')];assert len(closure)==1
    expected=('NEW_RESEARCH_CAPTURE_COMPLETE '+stamp+' '+str(len(om))+' '+str(len(ol))+'\n').encode()
    skip={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'};same=0;closed=[]
    for name,item in om.items():
        if name in skip:continue
        if name==closure[0] and item['bytes']==0 and m[name]!=item:
            assert z.read(name)==expected;closed.append(name);continue
        assert m[name]==item,'Old science changed: '+name;same+=1
    for name,item in ol.items():assert large[name]['sha256']==item['sha256'] and large[name]['bytes']==item['bytes']
proof=dict(status='MATCHED_MESSAGE_POOLS_'+a.phase.upper()+'_ORIGINALS_AND_ACTUAL_CAPTURE_VERIFIED',utc=rec['utc'],zip_sha256=rec['sha256'],members=len(m),required_original_files_verified=len(required),new_full_originals_with_capture_large_references=newlarge,prior_small_science_unchanged=same,prior_large_reference_unchanged=len(ol),prior_natural_stdout_closure=closed,old_fullweights_not_redownloaded=True,local_supplementary_coverage_not_declared_remote_by_this_audit=True)
a.out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))

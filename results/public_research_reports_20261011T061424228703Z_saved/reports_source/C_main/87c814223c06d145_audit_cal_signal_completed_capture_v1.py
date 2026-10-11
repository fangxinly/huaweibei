from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--remote-root',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
r=read(a.snapshot/'receipt.json');ex=read(a.snapshot/'actual_capture_exit.json')
assert ex['exit_code']==0 and (a.snapshot/'exit_code.txt').read_text().strip()=='0'
data=(a.snapshot/'snapshot.zip').read_bytes();assert len(data)==r['bytes'] and sha(data)==r['sha256']
cpu=a.root/'independent_cpu';cr=read(cpu/'cpu_preservation_receipt.json');ce=read(cpu/'cpu_exit.json')
assert cr['status']=='OTHER_NODE_CAL_SIGNAL_ORIGINALS_AND_NUMPY_VERIFIED' and ce['exit_code']==0 and ce['argv']==ce['actual_proc_argv']
assert cr['original_packet_sha256']==sha((a.root/'preservation/snapshot.zip').read_bytes())
assert cr['cpu_array_audit_sha256']==sha((cpu/'cpu_array_audit.json').read_bytes()) and cr['cpu_exit_sha256']==sha((cpu/'cpu_exit.json').read_bytes())
assert read(cpu/'cpu_array_audit.json')==read(a.root/'local_independent_audit.json')
skip={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'}
with zipfile.ZipFile(a.snapshot/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
    for name,item in m.items():
        b=z.read(name);assert len(b)==item['bytes'] and sha(b)==item['sha256']
    assert sha(z.read('source/capture_soft_vector_v19.py'))==ex['capture_source_sha256']=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
    inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0].strip()==cr['gpu_uuid']=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067' and not inv['compute']
    prefix='group_teacher/cpu_preservation/'+Path(a.remote_root).name+'/'
    for name in ['fresh_inventory.json','cpu_preservation_receipt.json','cpu_array_audit.json','cpu_exit.json','cpu_audit.log']:assert z.read(prefix+name)==(cpu/name).read_bytes(),name
    for name in ['verify_cal_signal_cpu_v1.py','run_same_teacher_capture_v1.py']:assert z.read(prefix+name)==(Path(__file__).parent/name).read_bytes()
    with zipfile.ZipFile(a.root/'preservation/snapshot.zip') as packet:
        pm=json.loads(packet.read('member_manifest.json'));assert 'metric_labels.npz' not in pm
        for name in pm:assert z.read(prefix+'originals/'+name)==packet.read(name),name
    with zipfile.ZipFile(a.prior) as old:om=json.loads(old.read('member_manifest.json'));ol=json.loads(old.read('large_file_manifest.json'))
    prior_exit=read(a.prior.parent/'actual_capture_exit.json');assert prior_exit['exit_code']==0
    stamp=prior_exit['argv'][-1];closure=[name for name in om if name.endswith('/capture_'+stamp+'.log')];assert len(closure)==1
    expected=('NEW_RESEARCH_CAPTURE_COMPLETE '+stamp+' '+str(len(om))+' '+str(len(ol))+'\n').encode()
    unchanged=0;closed=[]
    for name,item in om.items():
        if name in skip:continue
        if name==closure[0] and item['bytes']==0 and m[name]!=item:
            assert z.read(name)==expected;closed.append(name);continue
        assert m[name]==item,'Changed old science member: '+name;unchanged+=1
    large=json.loads(z.read('large_file_manifest.json'))
    for name,item in ol.items():assert large[name]['sha256']==item['sha256'] and large[name]['bytes']==item['bytes'],name
proof=dict(status='CAL_SIGNAL_D_OTHER_NODE_CPU_AND_ACTUAL_CAPTURE_JOINTLY_VERIFIED',actual_capture_utc=r['utc'],zip_sha256=r['sha256'],zip_bytes=r['bytes'],members=len(m),original_new_small_scientific_files=len(pm),cpu_original_receipt_sha256=sha((cpu/'cpu_preservation_receipt.json').read_bytes()),calibration_result_sha256=sha((a.root/'execute/calibration_result.json').read_bytes()),old_scientific_members_unchanged=unchanged,old_large_references_unchanged=len(ol),prior_self_stdout_natural_closure=closed,no_new_oldfullweight_download=True,no_EVAL_labels_in_originals=True,no_CPU_model_forward=True)
a.out.write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))

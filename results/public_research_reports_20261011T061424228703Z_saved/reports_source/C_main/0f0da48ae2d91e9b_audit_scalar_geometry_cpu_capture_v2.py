"""Validate actual B atomic capture against downloaded original CPU receipts."""
from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--originals',type=Path,required=True);p.add_argument('--remote-root',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((a.snapshot/'receipt.json').read_text());ex=json.loads((a.snapshot/'actual_capture_exit.json').read_text())
assert ex['exit_code']==0 and (a.snapshot/'exit_code.txt').read_text().strip()=='0'
data=(a.snapshot/'snapshot.zip').read_bytes();assert len(data)==r['bytes'] and sha(data)==r['sha256']
cpu=a.originals/'independent_cpu';cr=json.loads((cpu/'cpu_preservation_receipt.json').read_text());ce=json.loads((cpu/'cpu_exit.json').read_text())
assert cr['status']=='OTHER_NODE_SCALAR_GEOMETRY_ORIGINALS_AND_NUMPY_VERIFIED' and ce['exit_code']==0
assert cr['gpu_uuid']=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'
assert cr['cpu_array_audit_sha256']==sha((cpu/'cpu_array_audit.json').read_bytes()) and cr['cpu_exit_sha256']==sha((cpu/'cpu_exit.json').read_bytes())
assert cr['original_gpu_receipt_sha256']==sha((a.originals/'execute/receipt.json').read_bytes())
packet=(a.originals/'snapshot.zip').read_bytes();assert sha(packet)==cr['original_packet_sha256']
skip={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'}
with zipfile.ZipFile(a.snapshot/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
    for name,item in m.items():
        b=z.read(name);assert len(b)==item['bytes'] and sha(b)==item['sha256']
    assert sha(z.read('source/capture_soft_vector_v19.py'))==ex['capture_source_sha256']=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
    inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0].strip()==cr['gpu_uuid'] and not inv['compute']
    prefix='group_teacher/cpu_preservation/'+Path(a.remote_root).name+'/'
    for name in ['cpu_array_audit.json','cpu_exit.json','cpu_audit.log','cpu_preservation_receipt.json']:
        assert z.read(prefix+name)==(cpu/name).read_bytes(),name
    assert z.read(prefix+'original_packet_receipt.json')==(a.originals/'package_receipt.json').read_bytes()
    for name in ['verify_scalar_geometry_cpu_v1.py','run_same_teacher_capture_v1.py']:
        assert z.read(prefix+name)==(Path(__file__).parent/name).read_bytes()
    with zipfile.ZipFile(a.originals/'snapshot.zip') as packet_zip:
        pm=json.loads(packet_zip.read('member_manifest.json'))
        large=json.loads(z.read('large_file_manifest.json'));new_large_refs=[]
        for name in pm:
            key=prefix+'originals/'+name;data=packet_zip.read(name)
            if key in m:
                assert z.read(key)==data,name
            else:
                assert name=='execute/geometry_frozen.npz'
                ref=large[key]
                assert ref['path']==a.remote_root+'/originals/'+name
                assert ref['bytes']==len(data) and ref['sha256']==sha(data)
                assert pm[name]['sha256']==sha(data)
                new_large_refs.append(dict(name=name,bytes=len(data),sha256=sha(data),full_original_in_actual_CPU_packet=True))
    with zipfile.ZipFile(a.prior) as old:
        om=json.loads(old.read('member_manifest.json'));ol=json.loads(old.read('large_file_manifest.json'))
    prior_exit=json.loads((a.prior.parent/'actual_capture_exit.json').read_text());assert prior_exit['exit_code']==0
    prior_stamp=prior_exit['argv'][-1]
    closure='group_teacher/correction_calibration_collection_v2_20261006T0414Z/capture_'+prior_stamp+'.log'
    expected=('NEW_RESEARCH_CAPTURE_COMPLETE '+prior_stamp+' '+str(len(om))+' '+str(len(ol))+'\n').encode()
    count=0;closed=[]
    for name,meta in om.items():
        if name in skip:continue
        if name==closure and meta['bytes']==0 and m[name]!=meta:
            assert z.read(name)==expected;closed.append(name);continue
        assert m[name]==meta,'Changed old scientific member: '+name;count+=1
    large=json.loads(z.read('large_file_manifest.json'))
    for name,item in ol.items():assert large[name]['sha256']==item['sha256'] and large[name]['bytes']==item['bytes'],name
result=dict(status='ACTUAL_B_SCALAR_GEOMETRY_CPU_CAPTURE_AND_OLD_FROZEN_EVIDENCE_VERIFIED',new_large_array_references_joined_to_actual_originals=new_large_refs,actual_capture_utc=r['utc'],zip_sha256=r['sha256'],zip_bytes=r['bytes'],members=len(m),original_scientific_files=len(pm),cpu_original_receipts_downloaded_and_matched=True,old_members_unchanged=count,prior_capture_self_stdout_natural_closure=closed,old_large_weight_sha_references_unchanged=len(ol),fullweights_new_download=False,no_cpu_model_forward=True,source_sha256=sha(Path(__file__).read_bytes()))
a.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

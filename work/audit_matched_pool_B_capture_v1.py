from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--remote-root',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_text())
cr=read(r/'independent_cpu/cpu_preservation_receipt.json');assert cr['status']=='OTHER_NODE_MATCHED_MESSAGE_POOLS_ORIGINALS_AND_EVAL_NUMPY_VERIFIED'
assert cr['original_packet_sha256']==sha((r/'preservation/snapshot.zip').read_bytes())
for phase in ['precheck','execute','evaluation']:
    ep=r/'independent_cpu'/('cpu_'+phase+'_exit.json');ex=read(ep);assert ex['exit_code']==0 and ex['argv']==ex['actual_proc_argv']
    assert cr['exits_sha256'][phase]==sha(ep.read_bytes())
    cp=r/'independent_cpu'/('cpu_'+phase+'_audit.json');assert cr['audits_sha256'][phase]==sha(cp.read_bytes())
    assert read(cp)==read(r/phase/'independent_audit.json')
rec=read(a.snapshot/'receipt.json');ex=read(a.snapshot/'actual_capture_exit.json');assert ex['exit_code']==0 and (a.snapshot/'exit_code.txt').read_text().strip()=='0'
b=(a.snapshot/'snapshot.zip').read_bytes();assert len(b)==rec['bytes'] and sha(b)==rec['sha256']
prefix='group_teacher/cpu_preservation/'+Path(a.remote_root).name+'/'
with zipfile.ZipFile(a.snapshot/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    m=json.loads(z.read('member_manifest.json'));large=json.loads(z.read('large_file_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
    for name,item in m.items():
        b=z.read(name);assert len(b)==item['bytes'] and sha(b)==item['sha256']
    inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0].strip()==cr['gpu_uuid']=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067' and not inv['compute']
    assert sha(z.read('source/capture_soft_vector_v19.py'))==ex['capture_source_sha256']=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
    linked=[]
    with zipfile.ZipFile(r/'preservation/snapshot.zip') as packet:
        pm=json.loads(packet.read('member_manifest.json'));assert 'metric_labels.npz' not in pm
        for name,item in pm.items():
            key=prefix+'originals/'+name
            if key in m:assert z.read(key)==packet.read(name)
            else:assert large[key]['sha256']==item['sha256'] and large[key]['bytes']==item['bytes'];linked.append(key)
    for p in (r/'independent_cpu').iterdir():
        if p.is_file() and p.suffix in ['.json','.log','.txt']:assert z.read(prefix+p.name)==p.read_bytes(),p.name
    for name in ['verify_matched_pool_cpu_v1.py','run_same_teacher_capture_v1.py']:assert z.read(prefix+name)==(Path(__file__).parent/name).read_bytes()
    with zipfile.ZipFile(a.prior) as old:om=json.loads(old.read('member_manifest.json'));ol=json.loads(old.read('large_file_manifest.json'))
    oldexit=read(a.prior.parent/'actual_capture_exit.json');assert oldexit['exit_code']==0
    stamp=oldexit['argv'][-1];closure=[name for name in om if name.endswith('/capture_'+stamp+'.log')];assert len(closure)==1
    expected=('NEW_RESEARCH_CAPTURE_COMPLETE '+stamp+' '+str(len(om))+' '+str(len(ol))+'\n').encode()
    skip={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'};same=0;closed=[]
    for name,item in om.items():
        if name in skip:continue
        if name==closure[0] and item['bytes']==0 and m[name]!=item:
            assert z.read(name)==expected;closed.append(name);continue
        assert m[name]==item,'Changed prior science: '+name;same+=1
    for name,item in ol.items():assert large[name]['bytes']==item['bytes'] and large[name]['sha256']==item['sha256']
proof=dict(status='MATCHED_MESSAGE_POOLS_D_ORIGINALS_OTHER_NODE_CPU_AND_ACTUAL_B_CAPTURE_JOINTLY_PASSED',utc=rec['utc'],zip_sha256=rec['sha256'],members=len(m),original_packet_files=len(pm),large_originals_references_linked_to_full_packet=linked,prior_scientific_small_unchanged=same,prior_large_refs_unchanged=len(ol),prior_natural_stdout_closure=closed,CPU_model_forward=False,original_GPU_prediction_sha256=cr['original_GPU_prediction_sha256'],EVAL_metrics_sha256=sha((r/'evaluation/receipt.json').read_bytes()))
a.out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))

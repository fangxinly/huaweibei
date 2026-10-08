from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--node',choices=['C','B'],required=True);p.add_argument('--remote-root',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_text())
receipt=read(a.snapshot/'receipt.json');ex=read(a.snapshot/'actual_capture_exit.json');assert ex['exit_code']==0 and (a.snapshot/'exit_code.txt').read_text().strip()=='0'
b=(a.snapshot/'snapshot.zip').read_bytes();assert len(b)==receipt['bytes'] and sha(b)==receipt['sha256']
plan=read(r/'plan.json');base='group_teacher/';prefix=base+('cpu_preservation/' if a.node=='B' else '')+Path(a.remote_root).name+'/'
with zipfile.ZipFile(r/'preservation/snapshot.zip') as packet:
    assert packet.testzip() is None;pm=json.loads(packet.read('member_manifest.json'))
    assert set(packet.namelist())==set(pm)|{'member_manifest.json'}
    for n,it in pm.items():assert sha(packet.read(n))==it['sha256'] and len(packet.read(n))==it['bytes']
    with zipfile.ZipFile(a.snapshot/'snapshot.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        m=json.loads(z.read('member_manifest.json'));large=json.loads(z.read('large_file_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
        for n,it in m.items():assert sha(z.read(n))==it['sha256'] and len(z.read(n))==it['bytes']
        uuid={'C':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa','B':'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'}[a.node]
        inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0].strip()==uuid and not inv['compute']
        assert sha(z.read('source/capture_soft_vector_v19.py'))==ex['capture_source_sha256']=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
        linked=[]
        for n,it in pm.items():
            key=prefix+('originals/' if a.node=='B' else '')+n
            if key in m:assert z.read(key)==packet.read(n)
            else:assert large[key]['sha256']==it['sha256'] and large[key]['bytes']==it['bytes'];linked.append(key)
        if a.node=='B':
            cr=read(r/'independent_cpu/cpu_preservation_receipt.json');cx=read(r/'independent_cpu/cpu_array_exit.json')
            assert cr['status']=='OTHER_NODE_CAL_NATIVE_RADIUS_ORIGINALS_NUMPY_VERIFIED' and cr['original_packet_sha256']==sha((r/'preservation/snapshot.zip').read_bytes())
            assert cx['exit_code']==0 and cx['argv']==cx['actual_proc_argv']
            assert cr['exit_sha256']==sha((r/'independent_cpu/cpu_array_exit.json').read_bytes())
            assert cr['array_audit_sha256']==sha((r/'independent_cpu/cpu_array_audit.json').read_bytes())
            assert read(r/'independent_cpu/cpu_array_audit.json')==read(r/'mechanism/independent_audit.json')
            for pp in (r/'independent_cpu').iterdir():
                if pp.is_file():assert z.read(prefix+pp.name)==pp.read_bytes()
            for name in ['verify_native_radius_cpu_v1.py','run_same_teacher_capture_v1.py']:assert z.read(prefix+name)==(Path(__file__).parent/name).read_bytes()
        with zipfile.ZipFile(a.prior) as old:om=json.loads(old.read('member_manifest.json'));ol=json.loads(old.read('large_file_manifest.json'))
        ox=read(a.prior.parent/'actual_capture_exit.json');assert ox['exit_code']==0;stamp=ox['argv'][-1]
        closure=[n for n in om if n.endswith('/capture_'+stamp+'.log')];assert len(closure)==1
        expected=('NEW_RESEARCH_CAPTURE_COMPLETE '+stamp+' '+str(len(om))+' '+str(len(ol))+'\n').encode()
        skip={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'};same=0;closed=[]
        for n,it in om.items():
            if n in skip:continue
            if n==closure[0] and it['bytes']==0 and m[n]!=it:assert z.read(n)==expected;closed.append(n);continue
            assert m[n]==it,'Prior science changed: '+n;same+=1
        for n,it in ol.items():assert large[n]['sha256']==it['sha256'] and large[n]['bytes']==it['bytes']
proof={'status':'CAL_NATIVE_RADIUS_'+a.node+'_D_ORIGINALS_AND_ACTUAL_CAPTURE_VERIFIED','utc':receipt['utc'],'zip_sha256':receipt['sha256'],'members':len(m),'originals':len(pm),'large_original_references_linked_to_full_D':linked,'prior_scientific_small_unchanged':same,'prior_large_unchanged':len(ol),'prior_natural_stdout_closure':closed,'CPU_originals_independently_verified':a.node=='B','no_new_EVAL_or_fit':True}
a.out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))

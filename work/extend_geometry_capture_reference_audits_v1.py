"""New auditors preserve all assertions and explicitly join large NPZ references."""
from pathlib import Path
root=Path(__file__).parent
src=(root/'audit_scalar_geometry_capture_v2.py').read_text(encoding='utf-8')
anchor='    for name in required:assert z.read(prefix+name)==(a.originals/name).read_bytes(),name'
assert src.count(anchor)==1
replacement='''    large=json.loads(z.read("large_file_manifest.json"));new_large_refs=[]
    for name in required:
        local=(a.originals/name).read_bytes()
        if prefix+name in m:
            assert z.read(prefix+name)==local,name
        else:
            assert name=="execute/geometry_frozen.npz"
            ref=large[prefix+name]
            assert ref["path"]==a.remote_root+"/"+name
            assert ref["bytes"]==len(local) and ref["sha256"]==sha(local)
            new_large_refs.append(dict(name=name,bytes=len(local),sha256=sha(local),full_original_separately_downloaded=True))'''
src=src.replace(anchor,replacement).replace('phase=a.phase,actual_capture_utc','phase=a.phase,new_large_array_references_joined_to_actual_D_originals=new_large_refs,actual_capture_utc')
(root/'audit_scalar_geometry_capture_v3.py').write_text(src,encoding='utf-8')
src=(root/'audit_scalar_geometry_cpu_capture_v1.py').read_text(encoding='utf-8')
anchor='        for name in pm:assert z.read(prefix+\'originals/\'+name)==packet_zip.read(name),name'
assert src.count(anchor)==1
replacement='''        large=json.loads(z.read('large_file_manifest.json'));new_large_refs=[]
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
                new_large_refs.append(dict(name=name,bytes=len(data),sha256=sha(data),full_original_in_actual_CPU_packet=True))'''
src=src.replace(anchor,replacement).replace('actual_capture_utc=r[\'utc\']','new_large_array_references_joined_to_actual_originals=new_large_refs,actual_capture_utc=r[\'utc\']')
(root/'audit_scalar_geometry_cpu_capture_v2.py').write_text(src,encoding='utf-8')
print('NEW_AUDITORS_LARGE_NPZ_REFERENCE_JOIN_READY')

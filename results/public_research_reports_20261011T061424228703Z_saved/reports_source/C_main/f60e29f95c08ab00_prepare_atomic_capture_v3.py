from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'capture_soft_vector_v2.py').read_text().replace('capture_soft_vector_v2.py','capture_soft_vector_v3.py')
s=s.replace("members={};large={};zpath=out/'snapshot.zip'", "members={};large={};final_path=out/'snapshot.zip';zpath=out/'snapshot.pending'")
s=s.replace("(out/'receipt.json').write_text", "zpath.replace(final_path)\n(out/'receipt.json').write_text")
(r/'capture_soft_vector_v3.py').write_text(s,encoding='utf-8')
(r/'audit_soft_snapshot_v3.py').write_text((r/'audit_soft_snapshot_v2.py').read_text(encoding='utf-8').replace('capture_soft_vector_v2.py','capture_soft_vector_v3.py'),encoding='utf-8')
print('ATOMIC_CAPTURE_READY_EXISTING_V2_SOURCE_PRESERVED')

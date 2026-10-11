"""Keep first completed local audit and add richer array audit as a new version."""
import json,hashlib,datetime
from pathlib import Path
base=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference_first10_actual_20261006T151954Z')
work=Path(__file__).resolve().parent
source=work/'audit_staged10_D_local_v1.py';s=source.read_text(encoding='utf-8')
begin=s.index("with np.load(raw/'run/out/selected_best_donor_mechanism.npz'")
end=s.index("record={'status'",begin)
old=s[:begin]+s[end:]
old=old.replace("'donor_original_arrays_independently_rebuilt':reconstructed,",'')
new=s.replace("BASE/'local_D_joint_audit.json'","BASE/'local_D_joint_audit_v2.json'")
(work/'audit_staged10_D_local_v2.py').write_text(new,encoding='utf-8')
previous=json.loads((base/'完整流单参考前10实际原件保存中.json').read_text(encoding='utf-8'))
previous.pop('current_stage');previous.pop('updated_utc')
current=json.loads((base/'local_D_joint_audit.json').read_text(encoding='utf-8'))
# The completed v2 receipt already produced names v1; retain it verbatim as evidence.
verbatim=base/'local_D_joint_audit_v2_original_receipt_before_source_versioning.json'
verbatim.write_bytes((base/'local_D_joint_audit.json').read_bytes())
(base/'local_D_joint_audit.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source.write_text(old,encoding='utf-8')
note={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'first_completed_receipt_recovered_from_untouched_original_copy':True,
 'original_receipt_actual_utc':previous['actual_utc'],'new_donor_array_audit_actual_utc':current['actual_utc'],
 'scientific_run_source_or_arrays_changed':False,
 'correction':'Local audit initially reran at same output path after adding array reconstruction; first receipt restored from intact earlier saved copy, second verbatim receipt retained and richer source/output separated as v2.'}
(base/'local_audit_version_correction.json').write_text(json.dumps(note,indent=2)+'\n')
print(json.dumps(note))

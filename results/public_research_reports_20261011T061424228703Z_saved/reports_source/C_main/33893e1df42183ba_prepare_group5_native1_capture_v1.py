import datetime as dt,json,pathlib,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import write
raw=bytes.fromhex((BASE/'work/group5_careflow1_native_archive_receipt.hex').read_text().strip());r=json.loads(raw)
assert r['all_member_SHA_CRC_unique_exact_set_passed'] and r['members']==49
root=BASE/'outputs'/('group5_careflow1_native_capture_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
(root/'remote_original_archive_receipt.json').write_bytes(raw)
write(BASE/'work/group5_careflow1_native_capture_pointer.json',dict(root=str(root),archive_name=P(r['archive']).name,method='careflow',fold=1,
 expected_SHA=r['archive_SHA'],bytes=r['bytes'],remote=str(P(r['archive']).parent).replace('\\','/'),remote_receipt_actual_UTC=r['actual_UTC']))
print(str(root),flush=True)

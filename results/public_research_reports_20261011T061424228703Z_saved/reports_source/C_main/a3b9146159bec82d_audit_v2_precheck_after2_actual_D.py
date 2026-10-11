from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_evidence_candidate_v1 import zip_audit, write, now

base = Path('D:/CodexBackups/selective_flow_20261003_1105/paired_official_prechecks_v2_actual_20261007T012116Z')
for method, expected in (
    ('minimal_fixed_F', '6a4537daa030b073ca5958c584c20785525e23d8bc6c090c3f31c6293ca87352'),
    ('careflow', '5f82876c0b6c644c5238e1121098296d9b1d2bec877e01d55fa6691d603725bd'),
):
    p = base / method / 'a/run/out/precheck_after2_full.pt'
    before = now()
    result = zip_audit(p, expected)
    result.update(actual_start_utc=before, actual_complete_utc=now(), physical_D_path=str(p), actual_download=True)
    dest = base / method / 'actual_after2_D_SHA_CRC_receipt.json'
    if dest.exists():
        raise ValueError('Do not replace actual original receipt')
    write(dest, result)
    print(method, result, flush=True)

import base64,datetime,hashlib,json,pathlib,zipfile
r=pathlib.Path(__file__).resolve().parent
read=json.loads(base64.b64decode(''.join((r/'restore_completed_read_original.b64').read_text().split())))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cap=json.loads(base64.b64decode(read['files']['capture_receipt.json']));p=r/'complete_public_runtime_restore_original.zip';assert p.stat().st_size==cap['archive_bytes'] and sha(p)==cap['archive_SHA'] and cap['natural_exit']==0
dest=r/'runtime_original';dest.mkdir()
with zipfile.ZipFile(p) as z:
 rows=json.loads(z.read('member_manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in rows}|{'member_manifest.json'}
 for v in rows:
  b=z.read(v['name']);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];d=dest/v['name'];assert d.resolve().is_relative_to(dest.resolve());d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(b)
result=json.loads((dest/'runtime_result.json').read_bytes());plan=json.loads((r/'official_new_N2_inference_protocol.json').read_bytes());assert result['versions']==plan['runtime_versions'] and result['original_candidate_both_modes_synthetic_passed']
exit=json.loads((dest/'natural_exit.json').read_bytes());assert exit['natural_exit']==0 and json.loads((dest/'offline_install_natural_exit.json').read_bytes())['natural_exit']==0
receipt=dict(status='N2_NEW_EXACT24_RUNTIME_ASSETS_CANDIDATE_QUALIFICATION_D_FULLY_VERIFIED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(p),archive_SHA=sha(p),archive_bytes=p.stat().st_size,member_count=len(rows),all_member_SHA_CRC_unique_passed=True,original_capture=cap,result=result,natural_exit=exit,SCP_41313_natural_exit=0,B_real_inference_not_yet_dispatched=True)
(r/'runtime_D_preservation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=receipt['status'],SHA=sha(p),members=len(rows))))

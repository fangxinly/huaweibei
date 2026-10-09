import ast,base64,datetime,hashlib,json,pathlib,shutil,zipfile
r=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
read=json.loads(base64.b64decode(''.join((r/'restore_failure_read_original.b64').read_text().split())));write(r/'restore_failure_read_original.json',read)
cap=json.loads(base64.b64decode(read['files']['capture_receipt.json']));a=r/'first_public_restore_failure_original.zip';assert a.stat().st_size==cap['archive_bytes'] and sha(a)==cap['archive_SHA'] and cap['natural_exit']==1
with zipfile.ZipFile(a) as z:
 rows=json.loads(z.read('member_manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in rows}|{'member_manifest.json'}
 for v in rows:assert len(z.read(v['name']))==v['bytes'] and hashlib.sha256(z.read(v['name'])).hexdigest()==v['sha256']
write(r/'first_public_restore_failure_D_receipt.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(a),archive_SHA=sha(a),archive_bytes=a.stat().st_size,all_member_SHA_CRC_unique_passed=True,original_capture_receipt=cap,diagnosis='Expected original file inventory omits self-describing asset_manifest.json; strict member-set equality rejected that one original manifest before asset extraction. No B inference, scoring, or model load.',local_preservation_SCP_88050_natural_exit=0))
new=r.parent/'batch5_B_inference_manifest_repair_prepared_20261009T174820Z';new.mkdir()
source=(r/'batch5_B_posttrain_v1.py').read_text()
old="assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(plan['common_member_SHA'])"
replacement="assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(plan['common_member_SHA'])|{'asset_manifest.json'}\n  assert {v['archive']:v['sha256'] for v in json.loads(z.read('asset_manifest.json'))['files']}==plan['common_member_SHA']"
assert old in source;source=source.replace(old,replacement)
source=source.replace("write(root/'coordinator_failure.json',dict(actual_UTC=utc(),type=type(e).__name__,message=str(e)));code=1","import traceback;write(root/'coordinator_failure.json',dict(actual_UTC=utc(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));code=1")
ast.parse(source);helper=new/'batch5_B_posttrain_v1.py';helper.write_text(source,encoding='utf8')
plan=json.loads((r/'official_new_N2_inference_protocol.json').read_bytes());plan['plan_path']=plan['plan_path'].replace('20261009T174412Z','20261009T174820Z');plan['restore_root']=plan['restore_root'].replace('20261009T174412Z','20261009T174820Z');plan['assets_root']=plan['assets_root'].replace('20261009T174412Z','20261009T174820Z');plan['coordinator_SHA']=sha(helper);plan['revision']='Only original public ZIP self-manifest handling corrected; previous actual failure retained. Original infer core/model/checkpoint/source/assets/runtime and loss/selection unchanged.';plan['previous_failed_restore_original']=json.loads((r/'first_public_restore_failure_D_receipt.json').read_bytes())
with zipfile.ZipFile('D:/CodexBackups/selective_flow_20261003_1105/new_p4_assets_20261005T1220Z/common_assets.zip') as z:
 assert set(z.namelist())==set(plan['common_member_SHA'])|{'asset_manifest.json'} and z.testzip() is None
 assert {v['archive']:v['sha256'] for v in json.loads(z.read('asset_manifest.json'))['files']}==plan['common_member_SHA']
 for n,h in plan['common_member_SHA'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
planfile=new/'official_new_N2_inference_protocol.json';write(planfile,plan);cap=new/'posttrain_manifest_repair_source.zip';rows=[]
with zipfile.ZipFile(cap,'x',zipfile.ZIP_DEFLATED) as z:
 for p in [helper,planfile,r/'first_public_restore_failure_D_receipt.json']:
  z.write(p,p.name);rows.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)))
 z.writestr('member_manifest.json',json.dumps(rows,indent=2))
with zipfile.ZipFile(cap) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for v in rows:assert hashlib.sha256(z.read(v['name'])).hexdigest()==v['sha256']
result=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(cap),archive_SHA=sha(cap),archive_bytes=cap.stat().st_size,plan_path=plan['plan_path'],plan_SHA=sha(planfile),source_SHA=sha(helper),all_member_SHA_CRC_unique_passed=True,actual_original_public_archive_all_member_SHA_CRC_unique_passed=True,not_inference_dispatched=True)
write(new/'prepared_source_D_receipt.json',result);print(json.dumps(result))

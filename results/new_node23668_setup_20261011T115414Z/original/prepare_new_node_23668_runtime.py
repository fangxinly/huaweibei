import ast,datetime as dt,hashlib,json,pathlib,shutil
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;work=BASE/'work'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
stamp=dt.datetime.now(dt.timezone.utc);tag=stamp.strftime('%Y%m%dT%H%M%SZ')
root=BASE/'outputs'/('new_node_23668_runtime_prepared_'+tag);root.mkdir()
node=json.loads((work/'new_node_23668_active.json').read_bytes())
pointer=json.loads((work/'group5_idle_public_runtime_pointer.json').read_bytes())
template=P(pointer['root'])/'prepared_runtime_template.json'
assert sha(template)==pointer['template_SHA']
plan=json.loads(template.read_bytes())
source=work/'new_node_23668_public_runtime.py';ast.parse(source.read_text())
shutil.copyfile(source,root/source.name)
remote='/root/autodl-tmp/ni_node23668_'+tag
common=P('D:/CodexBackups/selective_flow_20261003_1105/new_p4_assets_20261005T1220Z/common_assets.zip')
wheels=work/'second_lease_linux_wheels_20261006T1401Z.zip'
assert sha(common)==plan['common']['whole_SHA'] and common.stat().st_size==plan['common']['bytes']
assert sha(wheels)==plan['wheels']['whole_SHA'] and wheels.stat().st_size==plan['wheels']['bytes']
plan.update(status='NEW_NODE23668_ISOLATED_PUBLIC_RUNTIME_FROZEN',source_SHA=sha(source),GPU_UUID=node['GPU_UUID'],lease_end_UTC=node['conservative_lease_end_UTC'],
    stage_budget_seconds=3600,saving_budget_seconds=7200,local_space_capture_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    local_C_free_bytes=shutil.disk_usage('C:/').free,local_D_free_bytes=shutil.disk_usage('D:/').free,
    torch_wheel=json.loads((work/'new_node_23668_preflight/official_torch_wheel_reference.json').read_bytes()),
    new_runtime_root=remote+'/runtime',original_runtime_template_SHA=pointer['template_SHA'],
    existing_completed_native_once_not_repeated=True,task_training_inference_scores_not_authorized_by_runtime_receipt=True)
plan['common']['pretransport_path']=remote+'/common_assets.zip';plan['wheels']['pretransport_path']=remote+'/second_lease_linux_wheels_20261006T1401Z.zip'
write(root/'plan.json',plan)
result={'root':str(root),'remote_root':remote,'plan_SHA':sha(root/'plan.json'),'source_SHA':sha(source),'original_local_common':str(common),'original_local_wheels':str(wheels),'created_UTC':dt.datetime.now(dt.timezone.utc).isoformat()}
write(work/'new_node_23668_runtime_pointer.json',result)
print(json.dumps(result))

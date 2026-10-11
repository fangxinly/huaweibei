"""Freeze a separate exact-byte SCP transport after preserving the HTTP failure."""
import ast,datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC
def main():
 failure=json.loads((BASE/'work/group5_N1_runtime_failure_pointer.json').read_bytes())
 assert failure['receipt']['all_captured_original_bytes_matched_remote_SHA'] and not failure['receipt']['new_approval_rejection']
 assert failure['github']['commit'] and failure['proof']['all_member_SHA_CRC_unique_exact_set_passed']
 old=BASE/'work/group5_public_runtime_v1.py';src=old.read_text(encoding='utf8')
 start=src.index('def fetch(spec,path):');end=src.index('\n\ndef run(a):',start)
 newfetch="""def fetch(spec,path):
    # This stage accepts only exact public archives already delivered by SCP.
    source=Path(spec['pretransport_path'])
    if path.exists() or source.resolve()==path.resolve():raise FileExistsError('Fresh private restore destination required')
    if not source.is_file() or source.stat().st_size!=spec['bytes'] or sha(source)!=spec['whole_SHA']:
        raise ValueError('Pretransport original archive SHA/length differs')
    with source.open('rb') as inp,path.open('xb') as out:
        shutil.copyfileobj(inp,out,8*1024**2)
    if sha(path)!=spec['whole_SHA']:raise ValueError('Private original copy changed')
"""
 new=src[:start]+newfetch+src[end:]
 before=ast.parse(src);after=ast.parse(new)
 assert [ast.dump(n) for n in before.body if not isinstance(n,(ast.FunctionDef,)) or n.name!='fetch']==[ast.dump(n) for n in after.body if not isinstance(n,(ast.FunctionDef,)) or n.name!='fetch']
 assert 'urlopen' not in newfetch and 'pickle' not in newfetch and 'torch' not in newfetch
 now=dt.datetime.now(dt.timezone.utc);root=DC.parent/('g5_N1_prestaged_runtime_plan_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir();q=root/'payload';q.mkdir()
 name='group5_public_runtime_prestaged_v2.py';(q/name).write_text(new,encoding='utf8');(BASE/'work'/name).write_text(new,encoding='utf8')
 oldptr=json.loads((BASE/'work/group5_N1_public_runtime_pointer.json').read_bytes());p=json.loads((P(oldptr['root'])/'payload/plan.json').read_bytes())
 remote='/data/coding/g5_N1_prestaged_runtime_'+now.strftime('%Y%m%dT%H%M%SZ')
 p.update(source_SHA=digest(q/name),transport='SEPARATELY_FROZEN_UNCHANGED_PUBLIC_SCP_ARCHIVES',predecessor_failure_archive_SHA=failure['proof']['archive_SHA'],
  local_C_free_bytes=shutil.disk_usage('C:/').free,local_D_free_bytes=shutil.disk_usage('D:/').free,local_space_capture_UTC=now.isoformat())
 p['common']['pretransport_path']=remote+'/common_assets.zip';p['wheels']['pretransport_path']=remote+'/second_lease_linux_wheels_20261006T1401Z.zip'
 assert p['local_C_free_bytes']>200*1024**2 and p['local_D_free_bytes']>40*1024**2 and (dt.datetime.fromisoformat(p['lease_end_UTC'])-now).total_seconds()>p['stage_budget_seconds']+7200
 common=DC.parent/'new_p4_assets_20261005T1220Z/common_assets.zip';wheels=BASE/'work/second_lease_linux_wheels_20261006T1401Z.zip'
 for f,s in [(common,p['common']),(wheels,p['wheels'])]:assert f.stat().st_size==s['bytes'] and digest(f)==s['whole_SHA']
 write(q/'plan.json',p);h=digest(q/'plan.json')
 wrapper="import subprocess,json,pathlib,datetime\nr=pathlib.Path("+repr(remote)+")\na="+repr(['/data/miniconda/envs/torch/bin/python','-B',remote+'/'+name,'--plan',remote+'/plan.json','--plan-sha',h,'--root',remote+'/runtime'])+"\nwith (r/'stdout.log').open('xb') as f:\n c=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT)\n (r/'dispatch.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,argv=a)))\n code=c.wait()\n(r/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,natural_exit=code)))\nraise SystemExit(code)\n"
 ast.parse(wrapper);(q/'wrapper.py').write_text(wrapper,encoding='utf8');shutil.copyfile(P(__file__),q/P(__file__).name)
 write(q/'static_source_scope.json',dict(all_other_source_AST_identical=True,changed_function='fetch',HTTP_download_removed=True,
  no_new_task_array_or_label_decode=True,scientific_dispatch=0,private_offline_install_contract_unchanged=True,previous_failure_preserved=failure))
 proof=seal(q,q/'public_runtime_plan_original.zip');write(root/'D_plan_receipt.json',proof)
 publish(P(proof['archive']),proof['archive_SHA'],'group5-N1-prestaged-runtime-plan-'+proof['archive_SHA'][:12]+'.zip',root/'Release_plan_receipt.json')
 ptr=dict(root=str(root),remote=remote,plan_SHA=h,Release=json.loads((root/'Release_plan_receipt.json').read_bytes()),proof=proof,SSH_session=39320,scientific_training_dispatched=False,common_local=str(common),wheels_local=str(wheels))
 write(BASE/'work/group5_N1_prestaged_runtime_pointer.json',ptr);print(json.dumps(ptr),flush=True)
if __name__=='__main__':main()

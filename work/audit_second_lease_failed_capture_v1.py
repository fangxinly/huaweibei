import argparse,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--complete-initial',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];base=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_precheck_failure_actual_20261006T141445Z').resolve();d=base/'a';utc=datetime.datetime.now(datetime.timezone.utc)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
r=read(d/'capture_receipt.json');assert sha(d/'snapshot.zip')==r['sha256'] and (d/'snapshot.zip').stat().st_size==r['bytes']
assert read(d/'actual_capture20_exit.json')['capture_actual_exit_code']==0
with zipfile.ZipFile(d/'snapshot.zip') as z:
 names=z.namelist();assert z.testzip() is None and len(names)==len(set(names))==70
 manifest=json.loads(z.read('member_manifest.json'));large=json.loads(z.read('large_file_manifest.json'))
 assert set(names)==set(manifest)|{'member_manifest.json'}
 for n,m in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256'] and len(z.read(n))==m['bytes']
 inv=json.loads(z.read('inventory.json'));assert inv['gpu_uuid']=='GPU-53696803-875e-eec8-2231-29db63579891' and not inv['compute']
 assert inv['natural_exit']['exit_code']==1 and not inv['natural_exit']['actual_scientific_precheck_complete']
 assert inv['launch']['child_pid']==498 and inv['natural_exit']['child_pid']==498
 assert sha(d/'natural_exit.json')==hashlib.sha256(z.read('run/natural_exit.json')).hexdigest()
 assert hashlib.sha256(z.read('run/child_stderr.log')).hexdigest()==inv['natural_exit']['raw_stderr_sha256']
 assert b'ACTUAL_PRECHECK_TIME_OR_GPU_BUDGET_EXCEEDED: mixed_singleton' in z.read('run/child_stderr.log')
 ep=json.loads(z.read('source/deployment_execution_plan_v1.json'))
 for n,h in {**ep['source_sha256'],**ep['role_and_order_sha256']}.items():assert hashlib.sha256(z.read('source/'+n)).hexdigest()==h
 rt=json.loads(z.read('source/runtime_candidate_plan.json'))
 for n,h in rt['asset_sha256'].items():
  path='public_assets/'+n
  assert (large[path]['sha256'] if path in large else hashlib.sha256(z.read(path)).hexdigest())==h
 initial=large['run/out/clean_initial_full.pt']
 complete=False
 if a.complete_initial:
  pending=d/'clean_initial_full.pt.pending';final=d/'clean_initial_full.pt';path=final if final.exists() else pending
  assert path.stat().st_size==initial['bytes'] and sha(path)==initial['sha256']
  if path==pending:pending.rename(final)
  complete=True
 summary={'actual_utc':utc.isoformat(),'capture_actual_utc':r['actual_utc'],'capture_source':r['capture_source_sha256'],'snapshot_sha256':sha(d/'snapshot.zip'),'all_70_members_SHA_CRC_unique_source_role_asset_natural_exit_verified':True,'actual_GPU_attempted':True,'actual_GPU_precheck_passed':False,'failure_stage':'mixed_singleton budget gate','child_pid':498,'natural_child_exit_code':1,'wrapper_exit_code':read(d/'actual_wrapper_exit.json')['wrapper_exit_code'],'exact_peak_allocated_or_reserved_not_logged':True,'elapsed_launch_to_child_exit_seconds':(datetime.datetime.fromisoformat(inv['natural_exit']['actual_utc'])-datetime.datetime.fromisoformat(inv['launch']['actual_utc'])).total_seconds(),'complete_initial_weight_reference':initial,'initial_full_D_download_and_file_SHA_complete':complete,'other_node_CPU_initial_audit_complete':False,'post_two_step_full_checkpoint_saved':False,'formal100_started':False,'new_performance_scores':False}
 (base/('capture_failure_joint_audit_'+utc.strftime('%Y%m%dT%H%M%SZ')+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
 (root/'outputs/第二租期固定流首次预检失败最新.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 doc='第二租期固定流首次GPU预检实际失败\n\n实际UTC '+utc.isoformat()+f'''。A原child498/ wrapper497，fresh UUID/空compute/全部public与新source-role SHA/本地D38GB/人类新24h门控后实际调用；child自然exit1，wrapper wait exit1。mixed_singleton预算门控失败，launch到exit约{summary['elapsed_launch_to_child_exit_seconds']:.2f}秒，远低于1200s，依源码逻辑指向Torch全程6GiB allocated/reserved峰值门槛；具体两项峰值未持久记录，不能冒已测值、OOM或预检通过。

完整初始pt存在、两步后full与最终科学回执未生成，不冒完整INNER重放/完整梯度原数值/后状态donor机制通过，后续100未启动。调用栈显示两步阶段走到预算检查，但梯度与时间只在内存，不能当独立原证据。下一先保存原capture/初始pt，再另freeze只补阶段峰值与partial原回执的新计量源，保持目标/批次/6GiB门槛，不直接放宽或把batch32累积当等价。HVP未实现。

新capture20实际CAPTURE_COMPLETE/自然exit0 UTC14:15:18，D {base.as_posix()}/a/snapshot.zip 70成员全SHA/CRC/unique/新root/source/fullargv/raw exit核过；capture19旧源SHA保留引用且未在新根执行。3大文件fresh稳定inode SHA引用不是新下载。本轮initial full D下载/SHA完成={complete}，异节点CPU未完成；保存研究整体仍未完成。

三节点offline24 Linux wheels依赖恢复exit0/import/version核过；A原在线下载真实ReadTimeout、旧退出码未回填，B/C仅自己的包下载被SIGINT用于已校验同版本离线包，不停训练。旧科学和正负成绩保留。第九审视已读，无待返回；当前失败未在完整D/CPU原件保存前发新审视。
'''
 (root/'outputs/第二租期固定流首次预检实际失败与接续.md').write_text(doc,encoding='utf-8')
 print(json.dumps(summary))

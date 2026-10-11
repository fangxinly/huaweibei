from pathlib import Path
r=Path(__file__).resolve().parent
text=(r/'capture_soft_vector_v5.py').read_text(encoding='utf-8').replace('capture_soft_vector_v5.py','capture_soft_vector_v6.py')
needle=" tree(finite_root/'checks','finite_preflight')"
assert text.count(needle)==1
addition="""
 formal=Path('/data/coding/finite_task_risk_v1_deployment_20261005T1450Z')
 for path in sorted(formal.glob('*')):
  if path.is_file() and path.suffix in ['.py','.json','.npy']:add(path,'finite_formal_source/'+path.name)
 tree(formal/'run','finite_run')
 fin_launch=json.loads((formal/'launch.json').read_text())
 fin_proc=Path('/proc')/str(fin_launch['pid'])
 fin_inv={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':inv['gpu'],'compute':inv['compute'],'launch':fin_launch,'actual_process_argv':None,'actual_process_status':None,'data_disk_free':shutil.disk_usage('/data').free,'root':str(formal)}
 if (fin_proc/'cmdline').exists():
  fin_inv['actual_process_argv']=(fin_proc/'cmdline').read_bytes().decode().split('\\0')[:-1]
  fin_inv['actual_process_status']=(fin_proc/'status').read_text()
 if (formal/'exit.json').exists():fin_inv['exit']=json.loads((formal/'exit.json').read_text())
 encoded=json.dumps(fin_inv,indent=2).encode();z.writestr('finite_inventory.json',encoded);members['finite_inventory.json']={'bytes':len(encoded),'sha256':sha(encoded)}
"""
text=text.replace(needle,needle+addition)
(r/'capture_soft_vector_v6.py').write_text(text,encoding='utf-8')
text=(r/'audit_soft_snapshot_v5.py').read_text(encoding='utf-8').replace('capture_soft_vector_v5.py','capture_soft_vector_v6.py')
(r/'audit_soft_snapshot_v6.py').write_text(text,encoding='utf-8')
print('CAPTURE6_INCLUDES_NEW_FINITE_FORMAL_SOURCE_PROCESS_AND_RUN')

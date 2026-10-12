import ast,datetime as dt,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;B=P(__file__).resolve().parent.parent;W=B/'work';sys.path.insert(0,str(W))
from group5_release_transport_v1 import digest,write
now=dt.datetime.now(dt.timezone.utc);r=B/'outputs'/('node20631_official_queue_prepared_'+now.strftime('%Y%m%dT%H%M%SZ'));r.mkdir();src=r/'source';src.mkdir()
old=W/'polarity_intensity_factorized_aux_qualified_20261008T203626Z';origin={}
for p in old.glob('*.py'):
 shutil.copyfile(p,src/p.name);origin[p.name]=dict(original=str(p),SHA=digest(p))
for n in ['node20631_train_seed_v1.py','node20631_CPU_audit_v1.py','node20631_parallel_supervisor_v1.py','group5_release_transport_v1.py','node23668_remote_release_stream_v1.py']:
 shutil.copyfile(W/n,src/n);origin[n]=dict(original=str(W/n),SHA=digest(W/n))
for p in src.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.name)
q=json.loads((old/'qualified_precheck_plan.json').read_bytes());runtime=json.loads((B/'outputs/node20631_runtime_direct_staged_20261012T153611Z/plan.json').read_bytes())
p=dict(execution_enabled=False,protocol='OFFICIAL_FIXED_SPLIT_FIVE_NEW_SEEDS',human_confirmation='官方划分不变，五随机种子重复',seeds=[4,5,6,7,8],methods=['careflow','factorized_aux','regression_aux'],official_train_dev_IDs=q['official_train_dev_IDs'],official_TEST_IDs=q['official_role_IDs']['TEST'],official_ID_identity_SHA=q['official_ID_identity_SHA'],source_SHA={n:digest(src/n) for n in origin},asset_SHA=runtime['asset_SHA'],runtime_versions=runtime['runtime_versions'],GPU_UUID=runtime['GPU_UUID'],lease_end_UTC=runtime['lease_end_UTC'],per_job_budget_seconds=21600,new_job_minimum_free_bytes=12*1024**3,maximum_concurrent_jobs=3,GPU_free_for_new_job_MiB=7200,per_process_GPU_ceiling_bytes=10*1024**3,training_recipe=dict(epochs=100,batch=32,updates=4000,drop_last=1,eval_batches=[128,101],checkpoint_selection='official VAL batch-MSE strict earliest',test='frozen selected checkpoint once; historically seen'),admission='First real 40 updates before next process; GPU free >= max(7200MiB,previous_peak_reserved*1.2), cgroup RAM and disk and lease; no score used',full_model_preservation='512MiB original ranges via local credential relay, actual whole Release restore SHA/CRC/exactset required',scientific_execution_qualification=False)
assert len(p['official_train_dev_IDs']['train'])==1281 and len(p['official_train_dev_IDs']['dev'])==229 and len(p['official_TEST_IDs'])==685
write(r/'prepared_plan.json',p);write(r/'source_origin.json',origin);write(r/'source_AST_report.json',dict(actual_UTC=now.isoformat(),Python_files=len(origin),AST_passed=True,no_scientific_import_or_array_model_decode=True,training_not_dispatched=True))
(r/'实验协议.md').write_text('官方 TRAIN1281/VAL229/TEST685 保持不变。三个预先固定方法各用4、5、6、7、8种子，15组，属于五种子重复而非五折交叉验证。TRAIN优化，VAL选择100轮中最早最优状态，最终TEST固定五项只算一次。TEST历史已见，不称新盲测，不按TEST改队列或方案。每组batch32，作者原drop_last一行，100轮4000更新，VAL128/101批等权MSE。CaReFlow保留作者按批归一化；两候选保留原TRAIN统计和损失。先一组实测40更新，再依据资源逐渐增加至最多3进程。不以填满显存为目标改变批量。每组完整状态、独立CPU审核、原日志与来源入512MiB分片Release并整件实际还原。此包仅准备，运行环境、传输路线及实际派发仍须核原回执。\n',encoding='utf8')
z=r/'source_bundle.zip'
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as f:
 for pth in src.iterdir():f.write(pth,'source/'+pth.name)
 for n in ['prepared_plan.json','source_origin.json','source_AST_report.json','实验协议.md']:f.write(r/n,n)
ptr=dict(root=str(r),source=str(src),bundle=str(z),bundle_SHA=digest(z),prepared_plan_SHA=digest(r/'prepared_plan.json'),remote='/root/autodl-tmp/ni_node20631_20261012T152116Z/official_queue',training_dispatched=False,once_consumed=False)
write(W/'node20631_official_queue_pointer.json',ptr);print(json.dumps(ptr))

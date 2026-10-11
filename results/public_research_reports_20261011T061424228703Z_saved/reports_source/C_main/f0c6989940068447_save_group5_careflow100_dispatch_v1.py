"""Preserve actual original dispatch bytes and observed updates, not completion."""
import datetime as dt,hashlib,io,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

def main():
    pp=json.loads((BASE/'work/group5_formal_plan_pointer.json').read_bytes());root=P(pp['root']);meta=root/'dispatch_public';meta.mkdir()
    b=bytes.fromhex((BASE/'work/group5_dispatch_original_capture.hex').read_text())
    assert hashlib.sha256(b).hexdigest()=='fa570c00722ede807c20f4994a97372951bee24566b0e1188a710690446f5911'
    (meta/'dispatch_metadata_original.zip').write_bytes(b)
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==5
        for n in z.namelist():
            p=meta/'remote_original'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
    remote=meta/'remote_original';d=json.loads((remote/'dispatch.json').read_bytes());c=json.loads((remote/'out/construction.json').read_bytes());s=json.loads((remote/'actual_progress_snapshot.json').read_bytes())
    assert d['child']==c['pid']==2302 and d['plan_SHA']==pp['plan_SHA']
    assert c['clean_initial_state_SHA']=='44d9e43a7734b6497d639e1c86dc04163fb1f895c499d35c93e17533e6135152'
    assert c['initial_rng_SHA']=='23edd15b3c8c02ea1ce66693d213ed3050c768641ac2032297763a3287e2a18d'
    assert s['last_FIT_step']['updates']>=25 and not c['outer_labels_decoded']
    for p in (root/'payload').glob('*'):
        if p.suffix!='.zip':shutil.copyfile(p,meta/p.name)
    for n in ['group5_direct_runtime_v2.py','group5_epoch_resume_v1.py','freeze_group5_careflow100_v1.py','save_group5_careflow100_dispatch_v1.py']:
        shutil.copyfile(BASE/'work'/n,meta/n)
    q=dict(status='CAREFLOW_FOLD0_FORMAL100_ACTUAL_FIT_UPDATES_RUNNING',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        remote_root=pp['remote'],SSH_session=23105,child=d['child'],plan_SHA=pp['plan_SHA'],plan_Release=pp['Release'],
        dispatch_original_SHA=hashlib.sha256(b).hexdigest(),snapshot=s,formal_training_started=True,
        final_outputs_completed=0,total_final_outputs=25,whole_queue_dispatched=False,outer_scored=False,
        native_precheck_once_not_repeated=True,formal_once_consumed=True,
        stage_budget_seconds=16200,saving_reserve_seconds=7200,lease_end_UTC='2026-10-10T16:30:58+00:00',
        continuation='Observe healthy child without stopping it. Natural exit then independent CPU original-state audit, full original ZIP and real Release download restoration before outer scoring. No duplicate train or precheck once. Other method/fold native and composite qualification remain pending.')
    write(meta/'actual_dispatch_preservation.json',q)
    proof=seal(meta,meta/'complete_actual_formal_dispatch.zip');write(root/'D_dispatch_receipt.json',proof)
    publish(P(proof['archive']),proof['archive_SHA'],'group5-careflow0-dispatch-'+proof['archive_SHA'][:12]+'.zip',root/'Release_dispatch_receipt.json')
    github=publish_tree(meta,'results/group5_careflow0_formal100_dispatch_20261010','Start fresh CaReFlow fold0 full100 after original CPU and Release restore qualification')
    write(root/'GitHub_dispatch_receipt.json',github)
    state=json.loads((DC/'D_current_research_state.json').read_bytes());e=state['latest_human_TEST_selected_group5']
    e.update(status=q['status'],latest_formal100_actual=dict(root=str(root),receipt=q,preservation=proof,github=github),formal_training_started=True,final_outputs_completed=0)
    state.update(updated_at_utc=q['actual_UTC'],next_gate=q['continuation'],github_source=github,
        current_research_execution_blocker='No active approval blocker for this protocol after human continue; earlier failures retained as history. CaReFlow fold0 formal child2302 running; full25 outputs not complete.')
    sync(state);write(BASE/'work/group5_formal_dispatch_pointer.json',dict(root=str(root),receipt=q,proof=proof,github=github))
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:
        f.write('\n'+q['actual_UTC']+' CaReFlow fold0正式100轮child2302 UTC04:47:40实际dispatch，原源码/资产/新鲜资源/租期门过、新once消费，原公共初态和RNG与合格三步预检完全同SHA；UTC04:48:15已25次FIT更新，非仅准备。原启动ZIP/回执Release与D bare GitHub'+github['commit']+'保存，最终0/25、OUTER未评分、未派全队列；原预检与旧once不重做。\n')
    print(json.dumps(dict(root=str(root),github=github,archive_SHA=proof['archive_SHA'],child=d['child'],observed_updates=s['last_FIT_step']['updates']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

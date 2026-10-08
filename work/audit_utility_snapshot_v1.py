"""Read-only small capture audit. Completed whole weights require --stage complete."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,shlex,sys,zipfile
import numpy as np
DEP='inflow_utility_v4_deployment_20261005T0346Z'
ASSIGN={'a':('none',6143,'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3'),
        'b':('fixed',6464,'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d'),
        'c':('predicted',10379,'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')}
ORIGINAL=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen')
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'inflow_utility_v4_20261005T0346Z'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True)
    a.add_argument('--stage',choices=['live','complete'],required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--previous',type=Path)
    a.add_argument('--nodes',nargs='+',choices=list(ASSIGN),default=list(ASSIGN))
    for node in ASSIGN:a.add_argument('--checkpoint-'+node,type=Path)
    c=a.parse_args();assert not c.out.exists()
    assert len(set(c.nodes))==len(c.nodes)
    rows=[];sharedref=None;orderref=None
    sys.path.insert(0,str(ORIGINAL/'outputs/monitoring_tools'))
    from verify_careflow_five_seeds_v1 import metrics
    for node in c.nodes:
        mode,pid,uuid=ASSIGN[node]
        folder=c.directory/node;proof=load(folder/'proof.json')
        assert sha(folder/'snapshot.zip')==proof['archive_sha256']
        assert (folder/'snapshot.zip').stat().st_size==proof['archive_bytes']
        assert proof['capture_source_sha256']==sha(HERE/'capture_utility_v4.py')
        with zipfile.ZipFile(folder/'snapshot.zip') as z:
            members=json.loads(z.read('member_manifest.json'))
            assert members==proof['members']
            assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)|{'member_manifest.json'}
            for name,entry in members.items():
                raw=z.read(name);assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
            prefix=DEP+'/run_'+mode+'/'
            p=json.loads(z.read(prefix+'protocol.json'));h=json.loads(z.read(prefix+'history.json'))
            launch=json.loads(z.read(DEP+'/training_launch.json'));resources=json.loads(z.read('actual_resources.json'))
            checks=json.loads(z.read(DEP+'/check_'+mode+'/checks.json'))
            log=z.read(DEP+'/training.log').decode()
            assert 'Traceback' not in log
            assert all(resources[k]['returncode']==0 for k in ('gpu','compute','processes'))
            assert uuid in resources['gpu']['output'] and launch['pid']==pid
            assert p['name']=='inflow_utility_v4' and p['mode']==mode and p['gpu_uuid']==uuid
            assert p['seed']==91813 and p['epochs']==100 and p['total_updates']==4000 and p['updates_per_epoch']==40
            assert p['train_samples']==1281 and p['valid_samples']==229 and p['pretrained_tensors_verified']==198
            assert p['initial_model_sha256']=='2fbecdf63882b94317a7a6b8986e5edb66cb1b2679d7939a0530706f08da8af9'
            assert p['initial_flow_sha256']=='1c60592d3426320b61c67c3a739931703473bfe337d65bbc218884607f2b267b'
            assert p['batch_order_sha256']=='fb2341411057c1040ec4bf104bdd9edaf2c080a569e56a8bec6f5e0f84769b13'
            assert checks['initial_zero_feedback_equality_verified'] and checks['label_isolation_verified'] and checks['utility_gradient_detach_verified'] and checks['optimizer_change_verified'] and checks['warmup_check_steps']==3
            for name,digest in p['own_source_sha256'].items():
                assert sha(SOURCE/name)==digest and hashlib.sha256(z.read(DEP+'/'+name)).hexdigest()==digest
            assert p==load(HERE/'utility_checks'/node/'protocol.json')
            shared={k:v for k,v in p.items() if k not in ('mode','gpu_uuid')}
            if sharedref is None:sharedref=shared
            else:assert shared==sharedref
            orders=np.load(io.BytesIO(z.read(prefix+'batch_orders.npy')),allow_pickle=False)
            assert orders.shape==(100,1280) and orders.dtype==np.dtype('<i8')
            assert hashlib.sha256(orders.tobytes()).hexdigest()==p['batch_order_sha256']
            for i,order in enumerate(orders):
                assert len(np.unique(order))==1280 and order.min()>=0 and order.max()<1281
                assert hashlib.sha256(order.tobytes()).hexdigest()==p['batch_order_epoch_sha256'][i]
            if orderref is None:orderref=orders
            else:assert np.array_equal(orderref,orders)
            assert 0<len(h)<=100 and [row['epoch'] for row in h]==list(range(1,len(h)+1))
            best=float('inf');best_epoch=0
            for row in h:
                assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
                assert len(row['last_train_batch_utility_weights'])==6 and np.isfinite(row['last_train_batch_utility_weights']).all()
                assert min(row['last_train_batch_utility_weights'])>=0 and max(row['last_train_batch_utility_weights'])<=1
                if mode=='none':assert row['last_train_batch_utility_weights']==[0]*6 and row['context_norm']==0
                if mode=='fixed':assert row['last_train_batch_utility_weights']==[.5]*6
                if row['valid_mse']<best:best=row['valid_mse'];best_epoch=row['epoch']
                assert row['best_epoch']==best_epoch and row['best_valid_mse']==best
                expected=row['task_mse']+.02*row['flow_matching']+.01*row['cycle_reconstruction']+.05*row['unimodal_sentiment']+.01*row['variance_floor']+.025*row['pair_sentiment']+.01*row['utility_calibration']
                assert abs(expected-row['train_total'])<1e-6
                assert abs(row['unimodal_sentiment']-.5*(row['source_unimodal_sentiment']+row['stage1_observer_sentiment']))<1e-6
            if c.previous:
                oldproof=load(c.previous/node/'proof.json')
                assert sha(c.previous/node/'snapshot.zip')==oldproof['archive_sha256']
                with zipfile.ZipFile(c.previous/node/'snapshot.zip') as old:
                    previous=json.loads(old.read(prefix+'history.json'))
                    assert h[:len(previous)]==previous and len(h)>len(previous)
                    assert p==json.loads(old.read(prefix+'protocol.json'))
            row={'node':node,'mode':mode,'pid':pid,'captured_at':resources['captured_at'],'epochs':len(h),
                'gpu':resources['gpu']['output'],'remote_free_bytes':resources['disk_free_bytes'],
                'archive_sha256':proof['archive_sha256'],'members':len(members),'last_dev_batch_utility_weights':h[-1]['last_train_batch_utility_weights'],
                'best_epoch_so_far':best_epoch,'best_valid_mse_so_far':best,'full_checkpoint_verified':False}
            if c.stage=='live':
                assert len(h)<100,'Use complete stage and new whole checkpoint audit after completion'
                assert resources['compute']['output'].strip()
                ps=[line.split(None,2) for line in resources['processes']['output'].splitlines()]
                actual=[line for line in ps if line[0]==str(pid)]
                assert len(actual)==1 and shlex.split(actual[0][2])==launch['args']
                assert 'INFLOW_UTILITY_RUN_COMPLETE' not in log
            else:
                assert len(h)==100 and 'INFLOW_UTILITY_RUN_COMPLETE' in log
                assert not resources['compute']['output'].strip()
                assert not any(line.split() and line.split()[0]==str(pid) for line in resources['processes']['output'].splitlines())
                s=json.loads(z.read(prefix+'selection.json'));r=json.loads(z.read(prefix+'results.json'))
                assert s['epochs']==100 and s['best_epoch']==best_epoch and s['valid_mse']==best
                assert r['selection']==s and r['test_accessed'] is False and r['model_state_unchanged'] is True
                assert s['protocol_sha256']==members[prefix+'protocol.json']['sha256']
                cp=getattr(c,'checkpoint_'+node);assert cp and cp.is_file()
                digest=sha(cp);assert digest==s['checkpoint_sha256']
                excluded=[entry for entry in proof['excluded_weights'] if entry['path'].endswith('/'+DEP+'/run_'+mode+'/best.pt')]
                assert len(excluded)==1 and cp.stat().st_size==excluded[0]['bytes']
                with np.load(ORIGINAL/'outputs/repeat5_experiments/completed/careflow_seed128/predictions.npz',allow_pickle=False) as ref:yref=ref['valid_y']
                with np.load(io.BytesIO(z.read(prefix+'predictions.npz')),allow_pickle=False) as saved:
                    y=saved['valid_y'];on=saved['valid_pred'];off=saved['condition_off_pred']
                    assert on.shape==off.shape==y.shape==(229,) and np.array_equal(y,yref)
                    measured={'valid':metrics(on,y),'frozen_condition_off':metrics(off,y)}
                    for group,values in measured.items():
                        assert set(values)==set(r[group])
                        assert all(abs(v-r[group][k])<1e-6 for k,v in values.items())
                    assert abs(measured['valid']['author_batch_mse']-best)<1e-5
                    if mode=='none':assert np.max(np.abs(on-off))<1e-5
                    for key,width in [('own',3),('pair',6),('utility',6),('predicted_weights',6)]:
                        assert saved[key].shape==(229,width) and np.isfinite(saved[key]).all()
                    u=saved['utility'];w=saved['predicted_weights']
                    assert np.max(np.abs(u))<=1 and w.min()>=0 and w.max()<=1
                    assert np.allclose(w,1/(1+np.exp(-4*u)),atol=1e-7,rtol=1e-7)
                row.update(full_checkpoint_verified=True,checkpoint=str(cp.resolve()),checkpoint_bytes=cp.stat().st_size,
                    checkpoint_sha256=digest,selection=s,metrics=measured,files={name:members[prefix+name] for name in ('protocol.json','batch_orders.npy','history.json','selection.json','results.json','predictions.npz')})
            rows.append(row)
    report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'UTILITY_SELECTED_NODES_ACTUAL_RUNNING_SOURCE_PROTOCOL_ORDER_OBJECTIVE_VERIFIED' if c.stage=='live' else 'UTILITY_SELECTED_NODES_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED',
        'nodes_verified':c.nodes,'rows':rows,'limits':'last_train_batch_utility_weights raw key is actually last DEV batch after eval; full utility calibration analysis remains separate. Exploratory single seed with node binding; small captures omit full checkpoints; independent leased seven-file target verification separately required.'}
    with c.out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

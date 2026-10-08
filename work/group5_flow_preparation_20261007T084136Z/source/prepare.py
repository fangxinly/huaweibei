"""Create an immutable label-free preparation bundle. Never enables execution."""
import argparse
import ast
import json
import shutil
from pathlib import Path
import numpy as np
from fold_contract import sha,make_folds


def prepare(root,clock):
    if root.exists():raise ValueError('Fresh preparation root required')
    candidate=Path(__file__).resolve().parent
    workspace=candidate.parents[1]
    parent=workspace/'work/paired_official_fulltrain_v2_20261007T011543Z'
    identity=workspace/'work/paired_final_TEST_pipeline_frozen_20261007T043535Z/source/parents/input_identity_A.json'
    parent_plan=parent/'paired_fulltrain_execution_plan.json'
    if sha(parent_plan)!='8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb':
        raise ValueError('Original source parent differs')
    old=json.loads(parent_plan.read_text(encoding='utf-8-sig'))
    ids=json.loads(identity.read_text(encoding='utf-8-sig'))
    canonical=sum([ids[k] for k in ['train_ids','dev_ids','test_ids']],[])
    folds=make_folds(canonical)
    root.mkdir();source=root/'source';source.mkdir();parents=source/'parents';parents.mkdir()
    for p in candidate.glob('*.py'):shutil.copy2(p,source/p.name)
    # Only recursive Python imports required by the new constructor/adapter.
    needed={'legacy_flow_model','finite_single_token_reader_v1','encoder_adapter',
            'fixed_flow_components_candidate','minimal_fixed_flow_v2',
            'official_fulltrain_dev_guard_candidate','paired_fulltrain_session_candidate',
            'careflow_train_dev_author_components_candidate'}
    copied=set()
    while needed-copied:
        for name in sorted(needed-copied):
            p=parent/(name+'.py')
            if sha(p)!=old['source_sha256'][p.name]:raise ValueError('Original imported source differs: '+name)
            shutil.copy2(p,source/p.name);copied.add(name)
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module and (parent/(node.module+'.py')).is_file():
                    needed.add(node.module)
    shutil.copy2(identity,parents/identity.name);shutil.copy2(parent_plan,parents/parent_plan.name)
    split={'status':'POOLED_VIDEO_GROUP5_LABEL_FREE_SPLIT','canonical_row_ids':canonical,'folds':folds,
           'input_identity_parent_sha256':sha(identity),'labels_used_to_split':False,
           'original_role_rows':{k:len(ids[k]) for k in ['train_ids','dev_ids','test_ids']},
           'rows':len(canonical),'videos':len({s.split('[')[0] for s in canonical}),
           'outer_rule':'Largest video first; hash seed128 tie; greedy minimum row count',
           'inner_rule':'Nearest hash-prefix to15% of nonouter rows; video-isolated',
           'official_TEST_is_no_longer_a_holdout':True,'historical_all_original_roles_explored':True}
    def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write(source/'split.json',split)
    budget=[]
    for f in folds:
        n=f['rows']['fit'];rng=np.random.default_rng(np.random.SeedSequence([128,f['fold']]))
        orders=np.stack([rng.permutation(n) for _ in range(100)])
        np.save(source/f"orders_fold{f['fold']}.npy",orders,allow_pickle=False)
        budget.append(dict(fold=f['fold'],**f['rows'],epochs=100,batch=32,drop_last=False,
                           updates=100*((n+31)//32),tail_rows=n%32 or 32))
    for p in source.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    plan={'status':'GROUP5_SOURCE_SPLIT_PREPARED_NOT_EXECUTION_FROZEN','execution_enabled':False,
          'actualclock_preparation_utc':clock,'methods':['anchored_flow','careflow'],
          'single_candidate':'Source-supervised zero-gain residual flow, fixed coefficients, no sweep',
          'objective':{'final_mse':1.,'final_mae':.25,'source_mse':.5,'nonzero_sign_softplus':.1},
          'flow':{'euler_steps':2,'passes':1,'slots':8,'donor_scale':.125,'role_head_scale':.1,
                  'initial_gain':0.,'gain_activation':'tanh','risk_bound_claimed':False},
          'optimizer':{'name':'AdamW','base_lr':1e-5,'gain_lr':1e-3,'weight_decay':.01,'warmup_fraction':.1,
                       'gain_weight_decay':0.,'clip_norm':1.},
          'task_seed':128,'fold_budgets':budget,'inner_selection':'FP64 all-row MSE, strict min, earliest tie',
          'source_sha256':{p.name:sha(p) for p in sorted(source.glob('*.py'))},
          'split_sha256':sha(source/'split.json'),
          'order_sha256':{p.name:sha(p) for p in sorted(source.glob('orders_*.npy'))},
          'asset_sha256':old['asset_sha256'],'precheck_fold':0,
          'precheck_D_other_CPU_joint':{},'native_dependency_freeze_pending':True,
          'assigned_gpu_uuid':{'A':'GPU-53696803-875e-eec8-2231-29db63579891',
                               'B':'GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f',
                               'C':'GPU-417d3577-0525-788b-7296-0808a0f52012'},
          'remote_free_floor_bytes':8*1024**3,'max_gpu_peak_bytes':6*1024**3,
          'storage_estimate_not_actual_new_file_bytes':{'ten_final_full_states_bytes':37080000000,
                'two_full_precheck_states_bytes':5940000000,'additional_saving_margin_bytes':8000000000},
          'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00',
          'human_asset_and_lease_budget_reference':old['human_lease_provenance_reference'],
          'complete_storage_reservation_reference':None,'remaining_queue_execution_budget_seconds':None,
          'checkpoint_cadence':'Every10epochs plus completion; full model/Adam/scheduler/RNG/orders/stats/history/selected state',
          'score_rule':'All10 OUTER original predictions + full states + D/independent CPU/capture before one pooled OOF scoring',
          'global_score_token':'/data/coding/multimodal_flow_public_20261006T1341Z/locks/group5_'+clock.replace(' ','_').replace(':','')+'.score_once',
          'pooled_metric_rule':'Acc7/Acc2/support-weightedF1/MAE/Pearson on concatenated OOF, plus fold mean/std',
          'not_five_independent_seeds':True,'official_TEST_no_longer_holdout':True,
          'task_weights_reused':False,'new_training_started':False,'native_Torch_check_passed':False,
          'source_parents':{str(parent_plan):sha(parent_plan),str(identity):sha(identity)},
          'pending':['Native synthetic flow check','3-step paired fold0 precheck + full preservation/CPU',
                     'Fresh per-node UUID/fullargv/assets/runtime/space','Full10state permanent storage capacity',
                     'Measured whole queue + at least2h saving reserve','Separate execution freeze']}
    write(root/'preparation_protocol.json',plan)
    return {'root':str(root),'protocol_sha256':sha(root/'preparation_protocol.json'),
            'rows':len(canonical),'videos':split['videos'],'fold_budgets':budget,
            'sources':len(plan['source_sha256']),'execution_enabled':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--clock',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.clock),ensure_ascii=False,indent=2))

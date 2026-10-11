"""Frozen input-only TEST identity acquisition. No tokenizer/Torch/model/labels.

The pinned trusted whole pickle materializes all roles; r[1] is never indexed.
The separate final prediction/scoring protocol must be frozen after this stage.
"""
import argparse,datetime,hashlib,json,os,pickle,shutil,subprocess,sys
from pathlib import Path
import numpy as np

def require(v,msg):
    if not v:raise PermissionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def canon(x,allow_empty=False):
    if isinstance(x,bytes):x=x.decode('utf-8')
    require(isinstance(x,str) and (allow_empty or bool(x)),'Expected original UTF8 string ID/token')
    return x
def inspect_inputs(container):
    # Explicitly acquire only features and IDs. Do not validate/access label shape.
    roles={r:container[r] for r in ('train','dev','test')}
    require(len(roles['train'])==1281 and len(roles['dev'])==229 and len(roles['test'])>0,'Original role counts')
    ids={r:[canon(x[2]) for x in rows] for r,rows in roles.items()}
    for r in ids:require(len(ids[r])==len(set(ids[r])),'Duplicate role IDs')
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        require(not set(ids[a])&set(ids[b]),'Official roles overlap')
    prior_id_sha=hashlib.sha256(json.dumps({r:ids[r] for r in ('train','dev')},sort_keys=True,
                                         ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    require(prior_id_sha=='9462739ff690facc2c10eec2e72e79a02c1b7c0a6ee346e893716894ed690747',
            'Prior actual official TRAIN/DEV order changed')
    layouts={};h=hashlib.sha256()
    for record,row_id in zip(roles['test'],ids['test']):
        raw=record[0];require(len(raw)==3,'Original words/visual/acoustic tuple required')
        words=[canon(x,allow_empty=True) for x in raw[0]]
        header=json.dumps({'row_id':row_id,'words':words},ensure_ascii=False,separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'little'));h.update(header)
        layout={'words':len(words),'numeric_inputs':[]}
        for values in raw[1:]:
            array=np.asarray(values)
            require(array.ndim==2 and array.dtype.kind in 'fi' and np.isfinite(array).all(),
                    'Original finite numeric modality layout required')
            require(array.shape[0]==len(words),'Original word-modality alignment changed')
            meta={'shape':list(array.shape),'dtype':str(array.dtype)}
            encoded=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
            h.update(len(encoded).to_bytes(8,'little'));h.update(encoded)
            data=np.ascontiguousarray(array).tobytes()
            h.update(len(data).to_bytes(8,'little'));h.update(data)
            layout['numeric_inputs'].append(meta)
        key=json.dumps(layout,sort_keys=True);layouts[key]=layouts.get(key,0)+1
    return {'test_ids':ids['test'],'train_ids':ids['train'],'dev_ids':ids['dev'],
            'test_rows':len(ids['test']),'prior_official_train_dev_ID_sha256':prior_id_sha,
            'test_raw_input_sha256':h.hexdigest(),'test_original_input_layouts':layouts,
            'all_role_labels_read':False,'TEST_entry_indexed_for_inputs_and_ID_only':True,
            'trusted_whole_pickle_other_role_bytes_materialized':True,
            'target_scale_not_established_by_inputs_or_ID':True}

def frozen_gate(bundle,plan_sha):
    f=bundle/'input_identity_plan.json';require(sha(f)==plan_sha,'Exact frozen input identity plan')
    plan=json.loads(f.read_text(encoding='utf-8'))
    require(plan['status']=='OFFICIAL_TEST_IDENTITY_INPUT_ONLY_PROTOCOL_FROZEN' and
            plan['labels_enabled'] is False and plan['model_or_training_enabled'] is False and
            plan['TEST_prediction_or_score_enabled'] is False,'Input-only role scope required')
    for n,h in plan['source_sha256'].items():require(sha(bundle/n)==h,'Identity source changed')
    for method,spec in plan['qualified_fixed_parent_joints'].items():
        t=json.loads((bundle/spec['training_joint_file']).read_text(encoding='utf-8'))
        r=json.loads((bundle/spec['fresh_joint_file']).read_text(encoding='utf-8'))
        require(t['status']=='ACTUAL_PAIRED_METHOD_FULLTRAIN_GPU_D_OTHER_CPU_CAPTURE_JOINT_PASSED_NO_FINAL_TEST' and
                r['status']=='ACTUAL_PAIRED_FULLTRAIN_D_ORIGINAL_OTHER_CPU_AND_FRESH_PUBLIC_SELECTED_REPLAY_JOINT_PASSED_NO_FINAL_TEST',
                'Both original whole model CPU and fresh replay joints required')
        require(t['method']==r['method']==method and t['plan_sha256']==r['plan_sha256']==plan['original_training_plan_sha256'] and
                r['whole_training_joint_sha256']==plan['source_sha256'][spec['training_joint_file']],
                'Original parent source/identity linkage')
    require(set(plan['qualified_fixed_parent_joints'])=={'minimal_fixed_F','careflow'},'Both fixed parents required')
    return plan

def native_gate(plan,assets,node,root):
    raw={}
    commands={'gpu_UUID':['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
              'compute':['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],
              'fullargv':['ps','-eo','pid,ppid,args'],
              'space':['df','-B1',str(root)]}
    for name,cmd in commands.items():
        raw[name]=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout
    require(raw['gpu_UUID'].strip()==plan['allowed_node_UUID'][node] and not raw['compute'].strip(),
            'Fresh actual assigned node UUID/empty compute required')
    free=shutil.disk_usage(root).free;require(free>=4*1024**3,'Actual node save-space floor')
    remaining=(datetime.datetime.fromisoformat(plan['conservative_lease_end_UTC'])-
               datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    require(remaining>=plan['execution_budget_seconds']+7200,'Execution plus2h saving reserve')
    checked={n:sha(assets/n) for n in plan['asset_sha256']}
    require(checked==plan['asset_sha256'],'Original complete public assets/source bytes changed')
    return {'actual_utc':utc(),'native_raw':raw,'original_assets_checked_sha256':checked,
            'remote_free_bytes':free,'remaining_lease_seconds':remaining,
            'conservative_human_lease_not_platform_confirmation':True,
            'human_lease_provenance_reference':plan['human_lease_provenance_reference']}

def main():
    p=argparse.ArgumentParser()
    for n in ('root','bundle','assets'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=('A','B'),required=True)
    p.add_argument('--parent-identity',type=Path)
    p.add_argument('--parent-identity-sha')
    a=p.parse_args();plan=frozen_gate(a.bundle,a.plan_sha)
    require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_final_TEST_identity_'),
            'Separate original input-only root required')
    require((a.parent_identity is None)==(a.node=='A'),'Only B independently audits original A identity')
    out=a.root/'out';out.mkdir(exist_ok=False)
    direct=native_gate(plan,a.assets,a.node,a.root);write(out/'actual_native_preflight.json',direct)
    # Native/assets hashing must finish before opening the trusted input container.
    with (a.assets/'assets/mosi.pkl').open('rb') as f:container=pickle.load(f)
    identity=inspect_inputs(container);del container
    result=dict(identity,status='ACTUAL_OFFICIAL_TEST_INPUT_ONLY_IDENTITY_COMPLETE',
                actual_utc=utc(),pid=os.getpid(),fullargv=sys.argv,root=str(a.root),node=a.node,
                plan_sha256=a.plan_sha,source_sha256=plan['source_sha256'],
                native_preflight_sha256=sha(out/'actual_native_preflight.json'),
                TEST_prediction_or_model_forward_or_scoring_executed=False)
    if a.node=='B':
        require(a.parent_identity_sha and sha(a.parent_identity)==a.parent_identity_sha,'Original A identity exact bytes')
        original=json.loads(a.parent_identity.read_text(encoding='utf-8'))
        require(original['status']=='ACTUAL_OFFICIAL_TEST_INPUT_ONLY_IDENTITY_COMPLETE' and
                original['node']=='A' and original['plan_sha256']==a.plan_sha,'Original A fixed source qualification')
        for k,v in identity.items():require(original[k]==v,'Independent original CPU identity/input mismatch: '+k)
        result['original_A_identity_sha256']=a.parent_identity_sha
        result['independent_raw_input_ID_order_equality_passed']=True
    write(out/'actual_stage_receipt.json',result)
    print(json.dumps({'status':result['status'],'node':a.node,'test_rows':identity['test_rows'],
                      'actual_utc':result['actual_utc'],'receipt_sha256':sha(out/'actual_stage_receipt.json'),
                      'labels_read':False,'model_forward':False}),flush=True)

if __name__=='__main__':main()

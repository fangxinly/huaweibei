"""Separate CPU complete checkpoint/array audit; no model forward or refitting."""
import argparse,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root);out=root/'out'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensors_sha(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):
        assert v.device.type=='cpu' and torch.isfinite(v).all()
        x=v.contiguous();h.update(n.encode());h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()
exit=json.loads((root/'natural_exit.json').read_text());r=json.loads((out/'actual_precheck_receipt.json').read_text())
assert exit['exit_code']==0 and exit['natural_exit'] and exit['actual_scientific_precheck_complete']
assert r['optimizer_steps']==2 and r['tail_optimizer_steps']==0 and not r['inner_labels_read']
assert r['inner_dummy_label_replacement_max_error']==0 and r['strict_full_disk_replay_max_error']<=1e-6
counts={}
for name,key,scope,step in [('clean_initial_full.pt','initial_full','CLEAN_INITIAL_NO_PRECHECK',0),('after_two_steps_full.pt','after_two_steps_full','PRECHECK_TWO_STEPS_NOT_FORMAL',2)]:
    path=out/name;assert sha(path)==r[key]['file_sha256'] and path.stat().st_size==r[key]['bytes']
    payload=torch.load(path,map_location='cpu',weights_only=True);m=payload['metadata'];s=payload['model']
    assert m==r[key]['metadata'] and m['fold']==0 and m['seed']==91819 and m['scope']==scope and m['optimizer_steps']==step
    assert tensors_sha(s)==m['state_sha256']
    stat={n:v for n,v in s.items() if n.startswith(('dberta.v6_audio_','dberta.v6_visual_'))}
    assert tensors_sha(stat)==m['fit_statistics_sha256'] and len(s)==r[key]['tensors']
    counts[name]={'file_sha256':sha(path),'state_sha256':m['state_sha256'],'tensors':len(s),'elements':sum(v.numel() for v in s.values())}
    del payload,s
with np.load(out/'inner_precheck_predictions.npz',allow_pickle=False) as z:
    assert z['row_ids'].shape==(153,) and z['prediction'].shape==(153,) and np.isfinite(z['prediction']).all()
    assert str(z['model_state_sha256'])==r['after_two_steps_full']['metadata']['state_sha256']
mechanism=[]
for name,declared in zip(('donor_mechanism_clean_initial.npz','donor_mechanism_after_two_steps.npz'),r['donor_mechanism']):
    with np.load(out/name,allow_pickle=False) as z:
        assert all(np.isfinite(z[n]).all() for n in z.files)
        assert np.array_equal(z['0_first_state'],z['1_first_state'])
        maximum=lambda x:float(np.max(np.abs(x)))
        for suffix in ('prediction','terminal_state','context','feedback'):assert maximum(z['0_'+suffix]-z['2_'+suffix])<=1e-6
        pairs={'terminal_scalar_change_max':('prediction',),'second_euler_state_change_max':('terminal_state',),'context_change_max':('context',)}
        for k,(suffix,) in pairs.items():assert abs(maximum(z['0_'+suffix]-z['1_'+suffix])-declared[k])<1e-12
        mechanism.append({'file':name,'sha256':sha(out/name),'stage':declared['stage'],'context_change_max':declared['context_change_max'],'terminal_scalar_change_max':declared['terminal_scalar_change_max']})
proof={'status':'INDEPENDENT_CPU_FULL_STATE_SHA_METADATA_AND_ARRAYS_PASSED_NOT_MODEL_FORWARD','actual_utc':datetime.now(timezone.utc).isoformat(),'pid':__import__('os').getpid(),'argv':sys.argv,'original_receipt_sha256':sha(out/'actual_precheck_receipt.json'),'natural_exit_sha256':sha(root/'natural_exit.json'),'source_sha256':sha(__file__),'checkpoints':counts,'mechanism':mechanism,'GPU_used':False,'new_model_forward_or_scores':False}
Path(a.out).write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))

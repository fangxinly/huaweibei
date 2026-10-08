"""Only public pretrained text weights; aligned A/V inputs; pooled cache has no labels."""
import argparse,datetime,hashlib,json,os,pickle,sys
from pathlib import Path
import numpy as np
from contract import read,write,sha

def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha
    for name,h in p['source_sha256'].items():assert sha(a.bundle/name)==h
    for name,h in p['asset_sha256'].items():assert sha(a.assets/name)==h
    assert not a.out.exists();a.out.mkdir()
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import DebertaV2Model,DebertaV2Tokenizer
    torch.set_num_threads(2);torch.manual_seed(128);torch.cuda.manual_seed_all(128)
    split=read(a.bundle/'split.json');expected=split['canonical_row_ids']
    with (a.assets/'assets/mosi.pkl').open('rb') as f:container=pickle.load(f)
    # Whole original pickle materializes label bytes, disclosed. No label scalar
    # is indexed, converted, returned or forwarded by this feature stage.
    records=list(container['train'])+list(container['dev'])+list(container['test']);del container
    ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in records];assert ids==expected
    pretrained=a.assets/'assets/deberta-v3-base';tok=DebertaV2Tokenizer.from_pretrained(pretrained,local_files_only=True)
    model=DebertaV2Model.from_pretrained(pretrained,local_files_only=True).eval().requires_grad_(False)
    original=torch.load(pretrained/'pytorch_model.bin',map_location='cpu');matched=0
    for k,v in model.state_dict().items():
        key=k if k in original else 'deberta.'+k
        if key in original:assert torch.equal(v,original[key].to(v.dtype));matched+=1
    assert matched>=190;del original;model.cuda();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};cpu_rng=torch.get_rng_state().clone();gpu_rng=[v.clone() for v in torch.cuda.get_rng_state_all()]
    text=[];audio=[];vision=[];batch=16;seq=50
    with torch.no_grad():
        for start in range(0,len(records),batch):
            input_ids=[];mask=[];content=[];av=[];vv=[]
            for (words,v,audio_raw),unused_label,row in records[start:start+batch]:
                tokens=[];inv=[]
                for j,word in enumerate(words):
                    pieces=tok.tokenize(word);tokens.extend(pieces);inv.extend([j]*len(pieces))
                tokens=tokens[:seq-2];inv=inv[:seq-2];assert len(tokens)>0
                vi=np.asarray(v,dtype=np.float32)[inv];au=np.asarray(audio_raw,dtype=np.float32)[inv]
                assert np.isfinite(vi).all() and np.isfinite(au).all()
                vv.append(vi.mean(0));av.append(au.mean(0));n=len(tokens)+2
                input_ids.append(tok.convert_tokens_to_ids([tok.cls_token]+tokens+[tok.sep_token])+[0]*(seq-n));mask.append([1]*n+[0]*(seq-n));content.append([0]+[1]*len(tokens)+[0]*(seq-len(tokens)-1))
            m=torch.tensor(mask,device='cuda');valid=torch.tensor(content,device='cuda').bool();x=torch.tensor(input_ids,device='cuda')
            h=model(input_ids=x,attention_mask=m).last_hidden_state;pooled=(h*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
            text.extend(pooled.cpu().numpy());audio.extend(av);vision.extend(vv)
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items());assert torch.equal(cpu_rng,torch.get_rng_state());assert all(torch.equal(x,y) for x,y in zip(gpu_rng,torch.cuda.get_rng_state_all()))
    data=dict(row_ids=np.asarray(ids),text=np.asarray(text,dtype=np.float32),audio=np.asarray(audio,dtype=np.float32),vision=np.asarray(vision,dtype=np.float32));assert all(np.isfinite(data[k]).all() for k in ('text','audio','vision'))
    np.savez(a.out/'public_frozen_features.npz',**data)
    write(a.out/'actual_cache_receipt.json',dict(status='PUBLIC_FROZEN_ENCODER_LABEL_FREE_FEATURE_CACHE_COMPLETE',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,rows=len(ids),dimensions={k:list(data[k].shape) for k in ('text','audio','vision')},public_weight_SHA=p['asset_sha256']['assets/deberta-v3-base/pytorch_model.bin'],matched_public_tensors=matched,cache_SHA=sha(a.out/'public_frozen_features.npz'),cache_bytes=(a.out/'public_frozen_features.npz').stat().st_size,no_task_checkpoint_loaded=True,parameters_and_RNG_unchanged=True,whole_original_pickle_materialized=True,label_scalars_indexed=False,official_TEST_inputs_cached=True,original_TEST_no_longer_holdout_in_merged_CV=True))
    print(json.dumps(read(a.out/'actual_cache_receipt.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())

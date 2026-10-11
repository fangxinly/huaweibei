"""Read two small original members via immutable ZIP parts, no weights/labels."""
import hashlib,json,sys,zipfile
from pathlib import Path
from ingest_split_capture import Parts
base=Path(__file__).resolve().parents[1]
folder=Path('D:/CodexBackups/selective_flow_20261003_1105/official_anchored_upgrade_actual_20261008T053429Z/A_train')
out=base/'outputs/code_review_response_20261008T154000Z';out.mkdir(exist_ok=True)
with Parts([folder/'capture.part00',folder/'capture.part01']) as parts,zipfile.ZipFile(parts) as z:
    names=z.namelist();assert len(names)==len(set(names))
    manifest=json.loads(z.read('member_SHA.json'))
    target=next(n for n in names if n.endswith('/first_real_TRAIN_gradient.json'))
    final=next(n for n in names if n.endswith('/training_result.json'))
    gbytes=z.read(target);rbytes=z.read(final)
    assert hashlib.sha256(gbytes).hexdigest()==manifest[target] and hashlib.sha256(rbytes).hexdigest()==manifest[final]
    (out/'original_first_real_TRAIN_gradient.json').write_bytes(gbytes)
    (out/'original_training_result.json').write_bytes(rbytes)
    grads=json.loads(gbytes);result=json.loads(rbytes)
    info=result['metadata']['parameter_info']
    def group(name):
        if name=='dberta.own_flow.gain':return 'gain'
        if name.startswith('dberta.model.'):return 'encoder'
        for key in ('forward_fields','reader','role_head','message'):
            if name.startswith('dberta.own_flow.'+key+'.'):return key
        return 'other_encoder_or_decoder'
    grouped={}
    for name,l1 in grads.items():
        assert name in info
        groupdata=grouped.setdefault(group(name),dict(tensors=0,numel=0,nonzero_tensors=0,gradient_L1_sum=0.))
        groupdata['tensors']+=1;groupdata['numel']+=info[name]['numel'];groupdata['nonzero_tensors']+=int(l1!=0);groupdata['gradient_L1_sum']+=l1
    for v in grouped.values():v['mean_absolute_gradient_per_parameter']=v['gradient_L1_sum']/v['numel']
    receipt=dict(status='ORIGINAL_FIRST_REAL_TRAIN_GRADIENT_MEMBER_SHA_CRC_CHECKED',actualclock_before_read_UTC=sys.argv[1],
        archive_parts_read_only=[str(folder/'capture.part00'),str(folder/'capture.part01')],whole_archive_not_rehashed=True,
        gradient_member=target,gradient_SHA256=manifest[target],training_result_member=final,training_result_SHA256=manifest[final],
        groups=grouped,first_update_only=True,not_persistent_gradient_or_update_evidence=True,
        RMS_or_actual_Adam_update_not_saved_in_this_member=True,no_model_forward_or_scoring=True)
    (out/'gradient_review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],groups=grouped,gradient_SHA256=manifest[target],full_weights_or_labels_decoded=False),ensure_ascii=False))

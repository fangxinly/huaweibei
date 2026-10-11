"""Version frozen runtime implementation; v1 remains an unexecuted source."""
from pathlib import Path
p=Path(__file__).with_name('minimal_fixed_staged_training_v1.py')
s=p.read_text(encoding='utf-8')
s=s.replace("assert text.encoder._gradient_checkpointing_func.keywords=={'use_reentrant':False,'preserve_rng_state':True}","assert text.encoder._gradient_checkpointing_func.keywords=={'use_reentrant':False,'preserve_rng_state':True}\n    del text")
s=s.replace('def predict_inner():','def predict_inner(dummy=0.0):')
s=s.replace('batch[3]=torch.zeros_like(batch[3])','batch[3]=torch.full_like(batch[3],dummy)')
s=s.replace("best_pred=cp['best_prediction']","best_pred=np.asarray(cp['best_prediction'],dtype=np.float32)")
old="""    # Fresh-instance strict replay remains mandatory before future OUTER/head.
    receipt="""
new="""    # Save at the planned boundary before diagnostic branch updates. They are
    # isolated and never contribute to the 220/2200 formal optimizer steps.
    continuation_proof=None
    if args.phase=='stage10':
        from resume_next_update_check_v1 import verify_next_update
        continuation_proof=verify_next_update(session,output/'complete_resume_full.pt',
            orders[10,:32],args.asset_base,bundle,output,budget)
        # The diagnostic restores the exact saved last10 model, state and RNG.
        assert tensor_sha(session.model.state_dict())==state_digest
    journal=session.guard.journal
    # Release all GPU ownership before constructing a fresh public instance.
    del session
    torch.cuda.empty_cache()
    session=construct_public_candidate(args.asset_base,bundle)
    session.model.dberta.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    del session.clean_initial
    disk=torch.load(output/'selected_best_full.pt',map_location='cpu',weights_only=True)
    assert tensor_sha(disk['model'])==best_meta['state_sha256']
    session.model.load_state_dict(disk['model'],strict=True);del disk
    assert tensor_sha(session.model.state_dict())==best_meta['state_sha256']
    fresh_replay=predict_inner();relabel=predict_inner(7.0)
    fresh_error=float(np.max(np.abs(fresh_replay-np.asarray(best_pred))))
    label_error=float(np.max(np.abs(fresh_replay-relabel)))
    if fresh_error>1e-6 or label_error!=0 or stats_sha()!=plan['fit_statistics_sha256']:
        raise RuntimeError('FRESH_INSTANCE_FULL_DISK_REPLAY_OR_RELABEL')
    assert tensor_sha(session.model.state_dict())==best_meta['state_sha256']
    np.savez(output/'fresh_selected_best_inner_replay.npz',row_ids=session.guard.ids['inner'],
        prediction=fresh_replay,model_state_sha256=np.asarray(best_meta['state_sha256']))
    # Existing fixed FIT rows only: do not scan amplitudes, checkpoints or labels.
    from donor_terminal_mechanism_precheck_v1 import donor_terminal_check
    mechanism,arrays=donor_terminal_check(session,row_batch(plan['mechanism_fit_rows']),
                                        'shared10_selected_best' if args.phase=='stage10' else 'formal100_selected_best')
    np.savez(output/'selected_best_donor_mechanism.npz',**arrays)
    if mechanism['seconds']>120:raise RuntimeError('MECHANISM_TWO_MINUTE_BUDGET')
    receipt="""
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("'fresh_instance_replay_pending':True", "'fresh_instance_replay_pending':False,'fresh_instance_strict_full_disk_replay_error':fresh_error,'dummy_label_replacement_error':label_error,'next_update_continuation_check':continuation_proof,'donor_mechanism':mechanism")
s=s.replace("'original_guard_journal':session.guard.journal,'OUTER_input", "'original_guard_journal':journal,'fresh_instance_guard_journal':session.guard.journal,'OUTER_input")
target=p.with_name('minimal_fixed_staged_training_v2.py')
if target.exists():raise FileExistsError('Never overwrite frozen candidate')
target.write_text(s,encoding='utf-8')
compile(s,str(target),'exec')
print(str(target))

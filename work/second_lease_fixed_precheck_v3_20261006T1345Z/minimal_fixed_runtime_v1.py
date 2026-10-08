"""Source-only constructor candidate; no launch/optimizer step entry point.

Local AST compilation does not certify asset availability, CUDA construction,
gradients, budget, or checkpoint replay. Formal 100 and OUTER authorization are
deliberately absent. Runtime execution requires a separately verified precheck.
"""
import hashlib, importlib.util, json, math, pickle, sys
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np
import torch
from encoder_adapter import content_mask, encode_masked, fit_statistics, install
from minimal_fixed_flow_v2 import MinimalFixedFlow, objective
from role_guard_v1 import FoldZeroGuard, file_sha

REMOVED_AUTHOR = ('reflow_a', 'reflow_v', 'reflow_a_b', 'reflow_v_b',
                  'rf_a', 'rf_v', 'rf_a_b', 'rf_v_b', 'pooler')

def tensor_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((tuple(x.shape), str(x.dtype))).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()

def forward_fixed(self, input_ids, visual, acoustic, label_ids=None, input_mask=None):
    if input_mask is None:
        raise ValueError('EXPLICIT_MASK_REQUIRED')
    valid = content_mask(input_mask)
    hidden = self.model(input_ids, attention_mask=input_mask)[0]
    text = self.LayerNorm_l(self.proj_l(hidden)) * valid[..., None]
    def normalized(values, name):
        return ((values-getattr(self, 'v6_'+name+'_mean')) /
                getattr(self, 'v6_'+name+'_std')) * getattr(self, 'v6_'+name+'_active') * valid[..., None]
    audio = self.proj_a(normalized(acoustic, 'audio').transpose(1, 2)).permute(2, 0, 1)
    vision = self.proj_v(normalized(visual, 'visual').transpose(1, 2)).permute(2, 0, 1)
    audio = self.LayerNorm_a(encode_masked(self.transa, audio, valid).transpose(0, 1)) * valid[..., None]
    vision = self.LayerNorm_v(encode_masked(self.transv, vision, valid).transpose(0, 1)) * valid[..., None]
    prediction, first, losses, trace = self.own_flow(
        torch.stack([text, audio, vision], 1), valid,
        lambda x: self.predictor(self.fusion(x)), label_ids)
    self.last_first_prediction, self.last_losses, self.last_trace = first, losses, trace
    # No decoder_modules registration: no duplicate module ownership/optimizer.
    return prediction[:, None], prediction.new_zeros(()), prediction.new_zeros(())

def import_pinned(name, path, expected_sha):
    path = Path(path).resolve()
    if file_sha(path) != expected_sha:
        raise ValueError('HELPER_SOURCE_SHA_MISMATCH')
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise ValueError('HELPER_IMPORT_PATH_COLLISION')
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def construct_public_candidate(asset_base, bundle):
    """Only for future separately authorized execution; constructor has no CLI."""
    asset_base, bundle = Path(asset_base), Path(bundle)
    plan = json.loads((bundle/'runtime_candidate_plan.json').read_text(encoding='utf-8'))
    # All assets are checked before pickle deserialization or helper import.
    for relative, expected in plan['asset_sha256'].items():
        if file_sha(asset_base/relative) != expected:
            raise ValueError('PUBLIC_ASSET_OR_DATA_SHA_MISMATCH: '+relative)
    for relative, expected in plan['source_sha256'].items():
        if file_sha(bundle/relative) != expected:
            raise ValueError('NEW_BUNDLE_SOURCE_SHA_MISMATCH: '+relative)
    for relative, expected in plan['role_file_sha256'].items():
        if file_sha(bundle/relative) != expected:
            raise ValueError('ROLE_OR_ORDER_SHA_MISMATCH: '+relative)
    sys.path.insert(0, str(asset_base/'assets'))
    helpers = import_pinned('run_careflow', asset_base/'assets/run_careflow.py',
                            plan['asset_sha256']['assets/run_careflow.py'])
    control = import_pinned('run_control_baseline', asset_base/'assets/run_control_baseline.py',
                            plan['asset_sha256']['assets/run_control_baseline.py'])
    args = SimpleNamespace(repo=asset_base/'assets/CaReFlow', backbone=asset_base/'assets/deberta-v3-base',
                           epochs=100, seed=plan['seed'])
    author, arguments = helpers.load_author(args)
    author.set_random_seed(plan['seed'])
    # Source-pinned trusted container; never index DEV/TEST entries.
    container = pickle.load((asset_base/'assets/mosi.pkl').open('rb'))
    train = container['train']; del container
    ids = {r: np.load(bundle/(r+'_0.npy'), allow_pickle=False) for r in ('fit', 'inner', 'outer')}
    mapping = json.loads((bundle/'train_row_video_mapping.json').read_text(encoding='utf-8'))
    guard = FoldZeroGuard(train, ids, [r['video_id'] for r in mapping])
    fit_inputs = author.get_appropriate_dataset(guard.inputs_only(ids['fit'], 'fit'))
    inner_inputs = author.get_appropriate_dataset(guard.inputs_only(ids['inner'], 'inner'))
    stats = fit_statistics(fit_inputs)
    steps = 100 * math.ceil(len(fit_inputs)/32)
    model, temporary_optimizer, temporary_scheduler = author.prep_for_training(steps)
    public_match = helpers.pretrained_check(model, args.backbone)
    core = model.dberta
    for name in REMOVED_AUTHOR:
        if not hasattr(core, name):
            raise ValueError('EXPECTED_AUTHOR_MODULE_MISSING: '+name)
        delattr(core, name)
    core.own_flow = MinimalFixedFlow()
    install(core, stats)
    for encoder in (core.transa, core.transv):
        encoder.embed_positions._float_tensor.zero_()
    core.forward = MethodType(forward_fixed, core)
    core.to(author.DEVICE)
    del temporary_optimizer, temporary_scheduler
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer, scheduler = control.optimizer_for(author, model, steps)
    parameter_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {id(p) for p in model.parameters()}:
        raise ValueError('OPTIMIZER_PARAMETER_UNIQUE_COMPLETE_COVERAGE')
    if len(optimizer.state) != 0:
        raise ValueError('CLEAN_OPTIMIZER_HAS_PRECHECK_STATE')
    fit_supervised = author.get_appropriate_dataset(guard.fit_supervision(ids['fit']))
    initial = {'model': {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
               'torch_rng': torch.get_rng_state().clone(),
               'cuda_rng': [r.clone() for r in torch.cuda.get_rng_state_all()],
               'seed': plan['seed'], 'fold': 0,
               'source_plan_sha256': file_sha(bundle/'runtime_candidate_plan.json')}
    receipt = {'scope': 'constructor_only_no_optimizer_steps', 'temporary_author_optimizer_discarded': True,
               'public_match': public_match, 'state_sha256': tensor_sha(initial['model']),
               'initial_state_is_separate_from_precheck_state': True,
               'optimizer_steps': 0, 'removed_author_modules': list(REMOVED_AUTHOR),
               'parameters': sum(p.numel() for p in model.parameters()),
               'parameter_tensors': len(list(model.parameters())), 'guard_journal': guard.journal}
    return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler, guard=guard,
                           fit=fit_supervised, inner_inputs=inner_inputs, statistics=stats,
                           clean_initial=initial, construction_receipt=receipt, arguments=arguments)

def forward_batch(model, batch):
    ids, visual, acoustic, labels, mask = batch
    return model(ids, visual.squeeze(1), acoustic.squeeze(1), labels, mask)[0].view(-1)

def fit_objective(model, batch):
    prediction = forward_batch(model, batch)
    return objective(prediction, batch[3], model.dberta.last_losses)

def require_real_gradients(model):
    # Checking non-None/finite is necessary; zeros are reported honestly.
    report = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError('RETAINED_PARAMETER_GRADIENT_FAILURE: '+name)
        report[name] = {'l1': float(p.grad.detach().abs().sum()), 'elements': p.numel()}
    return report

if __name__ == '__main__':
    raise SystemExit('SOURCE_ONLY_CANDIDATE_NO_GPU_PRECHECK_OR_TRAINING_LAUNCH')

import hashlib
import torch
from encoder_adapter import content_mask, encode_masked
from minimal_fixed_flow_v2 import objective


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


def forward_batch(model, batch):
    ids, visual, acoustic, labels, mask = batch
    return model(ids, visual.squeeze(1), acoustic.squeeze(1), labels, mask)[0].view(-1)


def fit_objective(model, batch):
    prediction = forward_batch(model, batch)
    return objective(prediction, batch[3], model.dberta.last_losses)


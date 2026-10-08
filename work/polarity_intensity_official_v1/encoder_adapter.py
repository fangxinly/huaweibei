"""Train-fitted channel scaling, explicit padding attention and pooling."""
from types import MethodType
import torch
import torch.nn.functional as F


def content_mask(mask):
    mask = mask.bool().clone()
    length = mask.sum(1)
    mask[:, 0] = False
    mask.scatter_(1, (length - 1)[:, None], False)
    assert mask.any(1).all()
    return mask


def fit_statistics(dataset):
    valid = content_mask(dataset.tensors[4])
    statistics = {}
    for name, index in (("audio", 2), ("visual", 1)):
        raw = dataset.tensors[index].squeeze(1)
        values = raw[valid].double()
        mean = values.mean(0)
        std = values.std(0, unbiased=False)
        active = std >= 1e-6
        statistics[name] = {"mean": mean.float(), "std": std.clamp(min=1e-6).float(), "active": active}
    return statistics


def masked_attention(self, query, key, value, attn_mask=None):
    return F.multi_head_attention_forward(
        query, key, value, self.embed_dim, self.num_heads,
        self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
        self.add_zero_attn, self.attn_dropout, self.out_proj.weight, self.out_proj.bias,
        training=self.training, key_padding_mask=self._padding_mask,
        need_weights=False, attn_mask=attn_mask)


def encode_masked(encoder, inputs, valid):
    # Use the explicit token mask to construct positions; never infer padding
    # from the first projected floating-point channel.
    positions = encoder.embed_positions(valid.long()).transpose(0, 1)
    x = encoder.embed_scale * inputs + positions
    x = F.dropout(x, p=encoder.dropout, training=encoder.training)
    for layer in encoder.layers:
        layer.self_attn._padding_mask = ~valid
        x = layer(x)
        x = x * valid.T[..., None]
    if encoder.normalize:
        x = encoder.layer_norm(x)
    return x * valid.T[..., None]


def install(core, statistics):
    for name in ("audio", "visual"):
        for field, values in statistics[name].items():
            core.register_buffer("v6_" + name + "_" + field, values.clone())
    for encoder in (core.transa, core.transv):
        for layer in encoder.layers:
            layer.self_attn.forward = MethodType(masked_attention, layer.self_attn)


def forward_v6(self, input_ids, visual, acoustic, label_ids=None, input_mask=None):
    assert input_mask is not None
    valid = content_mask(input_mask)
    hidden = self.model(input_ids, attention_mask=input_mask)[0]
    text = self.LayerNorm_l(self.proj_l(hidden)) * valid[..., None]

    def normalize(values, name):
        values = (values - getattr(self, "v6_" + name + "_mean")) / getattr(self, "v6_" + name + "_std")
        return values * getattr(self, "v6_" + name + "_active") * valid[..., None]

    audio = self.proj_a(normalize(acoustic, "audio").transpose(1, 2)).permute(2, 0, 1)
    vision = self.proj_v(normalize(visual, "visual").transpose(1, 2)).permute(2, 0, 1)
    audio = self.LayerNorm_a(encode_masked(self.transa, audio, valid).transpose(0, 1)) * valid[..., None]
    vision = self.LayerNorm_v(encode_masked(self.transv, vision, valid).transpose(0, 1)) * valid[..., None]
    source = torch.stack([text, audio, vision], 1)
    self.own_flow.decoder_modules = (self.fusion, self.predictor)
    prediction, first, losses, trace = self.own_flow(source, valid,
        lambda x: self.predictor(self.fusion(x)), label_ids)
    self.last_first_prediction, self.last_losses, self.last_trace = first, losses, trace
    return prediction[:, None], prediction.new_zeros(()), prediction.new_zeros(())


def forward_batch(model, batch):
    # Raw aligned tensors; no released per-batch global min-max transform.
    ids, visual, acoustic, labels, mask = batch
    return model(ids, visual.squeeze(1), acoustic.squeeze(1), labels, mask)

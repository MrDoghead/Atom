import torch
import torch.nn as nn
from quant import fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3, quantize_gptq, quantize_tensor, quantize_tensor_channel_group

def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        args
    ):
        super().__init__()
        self.args = args
        self.register_buffer('weight', originalLayer.weight)
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        self.scales = None # torch.zeros((int(W.shape[0]/self.quantizer.channel_group), self.n_nonout))
        self.zeros = None
        self.keep_scales = None # (channel, 1)
        self.keep_zeros = None
        self.maxq = 0
        self.channel_group = 0

    @torch.no_grad
    def bmm_4bit(self, x, w):
        return torch.matmul(x, w)

    @torch.no_grad
    def bmm_8bit(self, x, w):
        return torch.matmul(x, w)

    @torch.no_grad
    def _forward(self, x, inp_scales_hi, inp_base_hi, inp_scales_lo, inp_base_lo, blocksize=128):
        # mix-precision matmul
        # assume x and w are both symmetrically quantized
        assert self.args.act_group_size == self.args.weight_group_size
        assert x.dim() == 2 or x.shape[0] == 1, "only support bs=1"
        bs, seqlen, hidden_dim = x.shape
        x = x.squeeze()
        W = self.weight.T
        n_nonout = hidden_dim - self.args.keeper
        inp_scales_lo = inp_scales_lo.reshape(seqlen, -1)
        inp_base_lo = inp_base_lo.reshape(seqlen, -1)

        y = torch.zeros((x.shape[0], W.shape[1]), dtype=torch.float16).to(x.device)
        for i1 in range(0, n_nonout, blocksize):
            i2 = min(i1 + blocksize, n_nonout)
            x_block = x[:, i1:i2]
            w_block = W[i1:i2, :]
            x_block_scales = inp_scales_lo[:, i1//blocksize].reshape(-1,1).to(torch.float32)
            w_block_scales = self.scales[:, i1].reshape(-1,1)
            if self.channel_group > 0:
                w_block_scales = w_block_scales.repeat(1, self.channel_group).reshape(-1,1)
            w_block_scales = w_block_scales.to(x_block_scales.device).to(x_block_scales.dtype)
            block_scales = torch.matmul(x_block_scales, w_block_scales.T) # fp16
            y_lo = self.bmm_4bit(x_block, w_block)
            y += y_lo * block_scales
        
        if self.args.keeper > 0:
            assert self.args.keeper_precision == 3, "only support int8 keeper"
            x_hi = x[:, n_nonout:]
            w_hi = W[n_nonout:, :]
            inp_scales_hi = inp_scales_hi.to(torch.float32)
            w_hi_scales = self.keep_scales.to(inp_scales_hi.device).to(inp_scales_hi.dtype)
            hi_scales = torch.matmul(inp_scales_hi, w_hi_scales.T) # fp16
            y_hi = self.bmm_8bit(x_hi.to(torch.float32), w_hi.to(torch.float32)) # fp32
            y += y_hi * hi_scales

        return y.reshape(bs, seqlen, hidden_dim)
        
    @torch.no_grad()
    def forward(self, x, 
                real_quant=False, 
                inp_scales_hi=None, 
                inp_base_hi=None, 
                inp_scales_lo=None, 
                inp_base_lo=None,
            ):
        if real_quant:
            # forward+dequant
            y = self._forward(x, inp_scales_hi, inp_base_hi, inp_scales_lo, inp_base_lo)
            if self.bias:
                y = y + self.bias
        else:
            y = torch.functional.F.linear(x, self.weight, self.bias) # y = x@w.T + b
        return y
    
    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def requant(self, blocksize=128, groupsize=-1):
        if self.args.wbits >= 16:
            return
        
        W = self.weight.data.clone()
        Q = torch.zeros_like(W)
        n_nonout = W.shape[1] - self.args.keeper
        for i1 in range(0, n_nonout, blocksize):
            i2 = min(i1 + blocksize, n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            for i in range(count):
                w = W1[:, i]

                if groupsize > 0:
                    if (i1 + i) % groupsize == 0:
                        scale = self.scales[:, i1+i].reshape(-1,1)
                        zero = self.zeros[:, i1+i].reshape(-1,1)
                q = quantize_gptq(
                    w.unsqueeze(1), scale.to(w.device), zero.to(w.device), self.maxq, self.channel_group, real_quant=True, offset=False
                ).flatten()
                Q1[:, i] = q
            Q[:, i1:i2] = Q1

        if self.args.keeper > 0:
            keep_w = self.weight[:, -self.args.keeper:].clone().contiguous()
        
        # Whether to keep outliers in FP8
        # for outliers, groupsize=0
        if self.args.keeper_precision > 0:
            assert self.args.keeper > 0, "Keeper must be greater than 0"
            assert self.args.keeper_precision ==3, "only support int8 keeper now"
            if self.args.keeper_precision == 1:
                return
            elif self.args.keeper_precision == 2:
                return 
            elif self.args.keeper_precision == 3:
                keep_w = torch.clamp(torch.round(keep_w / self.keep_scales) + self.keep_zeros, -128, 127)

        Q[:, -self.args.keeper:] = keep_w
        self.weight = Q.reshape(self.weight.shape).to(self.weight.data.dtype)
        del keep_w
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return

        if self.args.keeper > 0:
            saved_w = self.weight[:, -self.args.keeper:].clone().contiguous()
        
        # Whether to keep outliers in FP8
        if self.args.keeper_precision > 0:
            assert self.args.keeper > 0, "Keeper must be greater than 0"
            if self.args.keeper_precision == 1:
                saved_w = fake_quantize_quarter_E5M2(saved_w)
            elif self.args.keeper_precision == 2:
                saved_w = fake_quantize_quarter_E4M3(saved_w)
            elif self.args.keeper_precision == 3:
                saved_w = quantize_tensor(saved_w, n_bits=8, group_size=0, tiling=0, sym=True, exponential=False)

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = 0

        self.weight = quantize_tensor_channel_group(
            self.weight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling
        )

        if self.args.keeper > 0:
            self.weight[:, -self.args.keeper:] = saved_w
            del saved_w
    
    def reorder(self, in_reorder_index, out_reorder_index=None):
        if self.args.reorder == True:
            in_reorder_index = in_reorder_index.to(self.weight.device)
            self.weight = torch.index_select(self.weight, 1, in_reorder_index)
            if out_reorder_index is not None:
                out_reorder_index = out_reorder_index.to(self.weight.device)
                self.weight = torch.index_select(self.weight, 0, out_reorder_index)
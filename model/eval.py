import torch
import torch.nn as nn
from tqdm import tqdm
import fnmatch
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

from odk_story4.ApprxPICPyTorch import ApprxPICPyTorch

class MyDataset(Dataset):
    def __init__(self, samples, seq_len, n_gpu) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.samples = samples
        self.n_samples = samples.numel() // seq_len
        self.n_padded_samples = 0
        if self.n_samples % n_gpu != 0:
            self.n_padded_samples = n_gpu - self.n_samples % n_gpu
            self.samples = F.pad(self.samples, (0, self.n_padded_samples*seq_len), mode='constant', value=0)

    def __len__(self):
        return self.n_samples+self.n_padded_samples
    
    def __getitem__(self, index):
        data = self.samples[0, (index * self.seq_len):((index + 1) * self.seq_len)]
        label = self.samples[0, (index * self.seq_len):((index + 1) * self.seq_len)][1:]
        return data, label

def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

@torch.no_grad()
def llama_eval_parallel(model, testenc, dev):
    model = model.cuda()
    model.eval()
    seqlen = model.seqlen

    testenc = testenc.input_ids
    test_dataset = MyDataset(testenc, seqlen, torch.distributed.get_world_size())
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             sampler=DistributedSampler(test_dataset))
    
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)

    # model = torch.nn.parallel.DistributedDataParallel(model, 
    #                                                   device_ids=[0,1], 
    #                                                   output_device=0, 
    #                                                   broadcast_buffers=False,
    #                                                   find_unused_parameters=True)
    nlls = []
    with torch.no_grad():
        for inp, label in tqdm(test_loader, desc="Eval"):
            batch = inp.cuda()
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous() # 结果的前n个token, (1, 4095, 32000)
            shift_labels = label.cuda() # label的后n个token, (1, 4095)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float()

            gather_nll = [torch.zeros_like(neg_log_likelihood) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_nll, neg_log_likelihood)
            nlls.extend(gather_nll)
    nlls = nlls[:test_dataset.n_samples]
    ppl = torch.exp(torch.stack(nlls).sum() / test_dataset.n_samples)

    return ppl.item()

@torch.no_grad()
def llama_eval(model, testenc, dev):
    model.eval()
    testenc = testenc.input_ids
    nsamples = 4 #testenc.numel() // model.seqlen
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    ) # 三维，记录每个sample下的输入张量（二维）
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp # 二维，已经过emb
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev) # 二维切片
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # decodeLayer输出
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states) # llama输出
        shift_logits = lm_logits[:, :-1, :].contiguous() # 结果的前n个token, (1, 4095, 32000)
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:] # label的后n个token, (1, 4095)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() #* model.seqlen # 为什么乘seqlen？
        nlls.append(neg_log_likelihood) # 记录每个样本的nll
    #ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    ppl = torch.exp(torch.stack(nlls).sum() / nsamples)

    return ppl.item()

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()

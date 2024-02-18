import numpy as np
import torch

DEV = torch.device('cuda:0')

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train') # 36718行1列，每一行为一条文本
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt') # 切成一维tokens (1,2874559)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples): # 随机取128条数据
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j] # 随机抽取一段长度为`seqlen`的tokens
        tar = inp.clone()
        tar[:, :-1] = -100 # 只保留最后一个token，其他都赋值为-100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset, VerificationMode
    """
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    """
    traindata = load_dataset("json", data_files="/data/datasets/c4/en/c4-train.00000-of-01024.json.gz", split='train', verification_mode=VerificationMode.NO_CHECKS)
    valdata = load_dataset("json", data_files="/data/datasets/c4/en/c4-validation.00000-of-00008.json.gz", split='train', verification_mode=VerificationMode.NO_CHECKS)
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test') # diff1: validation -> test
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt') #diff2: "\n\n" -> " "
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset, VerificationMode
    """
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    """
    traindata = load_dataset("json", data_files="/data/datasets/c4/en/c4-train.00000-of-01024.json.gz", split='train', verification_mode=VerificationMode.NO_CHECKS)
    valdata = load_dataset("json", data_files="/data/datasets/c4/en/c4-validation.00000-of-00008.json.gz", split='train', verification_mode=VerificationMode.NO_CHECKS)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    assert "llama" in model.lower(), "Only llama models are supported."

    from transformers import LlamaTokenizer 
    tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    # Fix for transformer 4.28.0.dev0 compatibility
    # See: https://github.com/Vahe1994/SpQR/blob/main/datautils.py#L164
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        try:
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
        except AttributeError:
            pass
            print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
    
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model, tokenizer)
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, tokenizer)
        return get_c4(nsamples, seed, seqlen, model, tokenizer)

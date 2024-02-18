import torch
from quant import *
from outlier import *
from eval import *
from collections import defaultdict
from pprint import pprint
from modelutils_llama import quantize_model_llama, reorder_model_llama, quantize_model_gptq_llama,  add_act_quant_wrapper_llama
from modelutils_opt import quantize_model_opt, reorder_model_opt, quantize_model_gptq_opt,  add_act_quant_wrapper_opt
from parallel_utils import map_layers_to_multi_gpus
from LMClass import LMClass
import os
from transformers import LlamaForCausalLM
import argparse
from datautils import *
from eval import pattern_match
from lm_eval import tasks as lm_tasks
from lm_eval import evaluator as lm_evaluator

def get_llama(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    print(f"Load fp16 model from {model}")
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    #model.seqlen = 2048
    model.seqlen = 4096 # llama2
    return model

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, 
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    # Quantization Method
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantizing weight; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantizing activation; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--exponential', action='store_true',
        help='Whether to use exponential-only for weight quantization.'
    )
    parser.add_argument(
        '--a_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--w_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Whether to perform static quantization (For activtions). Default is dynamic. (Deprecated in Atom)'
    )
    parser.add_argument(
        '--weight_group_size', type=int, default=0, choices=[0, 32, 64, 128, 256, 384, 768],
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=int, default=0, choices=[0, 64, 128, 256, 384, 768],
        help='Group size when quantizing activations. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--reorder', action='store_true',
        help='Whether to keep salient weight unquantized.'
    )
    parser.add_argument(
        '--act_sort_metric', type=str, default='hessian', choices=['abs_mean', 'hessian'],
        help='The metric used to sort the activations.'
    )
    parser.add_argument(
        '--keeper', type=int, default=0,
        help='Group size to keep outliers.'
    )
    parser.add_argument(
        '--keeper_precision', type=int, default=0, choices=[0, 1, 2, 3],
        help='Precision to keep outliers. 0 for FP16; 1 for E5M2; 2 for E4M3; 3 for INT8 Quant.'
    )
    parser.add_argument(
        '--cache_index', action='store_true',
        help='Whether to use cached reorder index'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity (Deprecated in Atom).'
    )
    parser.add_argument(
        '--kv_cache', action='store_true',
        help='Whether to quant KV_Cache'
    )
    parser.add_argument(
        '--use_gptq', action='store_true',
        help='Whether to use GPTQ for weight quantization.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--w_clip_ratio', type=float, default=1.0,
        help='Clip ratio for weight quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--kv_clip_ratio', type=float, default=1.0,
        help='Clip ratio for kv cache quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help='Whether to evaluate perplexity.'
    )
    parser.add_argument(
        "--eval_common_sense", action="store_true",
        help='Whether to evaluate zero-shot accuray on commonsense reasoning tasks.'
    )
    parser.add_argument(
        "--multigpu", action="store_true", 
        help="at eval, map model to multiple gpus"
    )
    parser.add_argument(
        "--lm_eval_num_fewshot", type=int, default=0, 
        help="Number of shots in lm evaluation. Default is 0 for zero-shot."
    )
    parser.add_argument(
        "--lm_eval_limit", type=int, default=-1, 
        help="Limit the number of examples in lm evaluation"
    )
    parser.add_argument(
        '--save_dir', type=str, default='./saved',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--load_qmodel', type=str, default='',
        help='Path to load the quantized model.'
    )
    parser.add_argument(
        '--eval_prompt', action="store_true",
        help='Whether to evaluate prompt.'
    )
    parser.add_argument(
        '--prompt', type=str, default='LLM is',
        help='must set --eval'
    )
    
    args = parser.parse_args()
    print("args:", args)

    model_name = args.model.lower().split('/')[-1]
    assert model_name != None, "Please check the model path."

    if "llama" in args.model.lower():
        model = get_llama(args.model)
        get_act_stats_func = get_act_stats_llama
        reorder_model_func = reorder_model_llama
        add_act_quant_wrapper_func = add_act_quant_wrapper_llama
        quantize_model_gptq_func = quantize_model_gptq_llama
        quantize_model_func = quantize_model_llama
        eval_func = llama_eval
    elif "opt" in args.model.lower():
        model = get_opt(args.model)
        get_act_stats_func = get_act_stats_opt
        reorder_model_func = reorder_model_opt
        add_act_quant_wrapper_func = add_act_quant_wrapper_opt
        quantize_model_gptq_func = quantize_model_gptq_opt
        quantize_model_func = quantize_model_opt
        eval_func = opt_eval
    model.eval()

    if args.reorder:
        reorder_index_file_path = f'{args.save_dir}/{model_name}_reorder_index_{args.dataset}.pt'
        if args.cache_index == False:
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Getting activation stats...")
            act_scales = get_act_stats_func(
                model, dataloader, DEV, metric=args.act_sort_metric
            )

            print("Getting reording index...")
            reorder_index = get_reorder_index(model, act_scales)

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(reorder_index, reorder_index_file_path)
            print(f"reorder_index saved at {reorder_index_file_path}")
        else:
            assert os.path.isfile(reorder_index_file_path), "reorder index file not found."

            print(f"Loading cached reording index from {reorder_index_file_path}")
            reorder_index = torch.load(reorder_index_file_path)

        print("Reordering model...")
        model = reorder_model_func(
            model, device=DEV, args=args, reorder_index=reorder_index
        )

    if args.static == True:
        assert args.abits < 16, "Static quantization should quantize A."
        if args.cache_index == False:
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Getting scales...")
            scales = get_act_scales(model, dataloader, DEV, args)
            scales_save_path = f'../saved/{model_name}_scales_{args.dataset}_ags{args.act_group_size}_kp{args.keeper}_kpp{args.keeper_precision}.pt'
            torch.save(scales, scales_save_path)
            print("scales saved at: {scales_save_path}")
        else:
            print("Getting cached scales...")
            scales = torch.load(f'../saved/{model_name}_scales_{args.dataset}_ags{args.act_group_size}_kp{args.keeper}_kpp{args.keeper_precision}.pt')
    else:
        scales = defaultdict(lambda: None)


    if args.load_qmodel == '':
        if args.abits < 16:
            print("Inserting activations quantizers ...")
            scales = defaultdict(lambda: None)
            model = add_act_quant_wrapper_func(model, device=DEV, args=args, scales=scales)

        if args.wbits < 16:
            print("Quantizing...")
            if args.use_gptq:
                dataloader, testloader = get_loaders(
                    args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
                )
                model = quantize_model_gptq_func(model, device=DEV, args=args, dataloader=dataloader)
            else:
                model = quantize_model_func(model, device=DEV, args=args)
        # save model
        if args.save_dir:
            print(f"full qmodel is saved at {args.save_dir}/")
            torch.save(model, f'{args.save_dir}/{model_name}_w{args.wbits}a{args.abits}_{args.dataset}.pt')
    else:
        print(f"load qmodel from {args.load_qmodel}")
        model = torch.load(args.load_qmodel)
    
    if args.eval_ppl:
        datasets = ['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_func(model, testloader, DEV)

            print(f"targetResult,{dataset},{ppl:.3f}")
    
    # eval zero shot accuracy on commonsense datasets
    if args.eval_common_sense:
        lm = LMClass(args, model)
        lm.seqlen = 4096 # llama2 
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False

        if args.multigpu:
            if "llama" in args.model.lower():
                map_layers_to_multi_gpus(lm.model.model.layers)
                input_device = lm.model.model.layers[0].device
                output_device = lm.model.model.layers[-1].device
                assert input_device == output_device
                lm._device = input_device
                lm.model.model.embed_tokens.to(input_device)
                lm.model.model.norm.to(output_device)
                lm.model.lm_head.to(output_device)
            elif "opt" in args.model.lower():
                map_layers_to_multi_gpus(lm.model.model.decoder.layers)
                input_device = lm.model.model.decoder.layers[0].device
                output_device = lm.model.model.decoder.layers[-1].device
                assert input_device == output_device
                lm._device = input_device
                lm.model.model.decoder.embed_tokens.to(input_device)
                lm.model.model.decoder.embed_positions.to(input_device)
                lm.model.model.decoder.final_layer_norm.to(input_device)
                lm.model.lm_head.to(output_device)
        else:
            lm._device = DEV
            lm.model = lm.model.to(lm.device)

        results = {}
        tasks_str = "piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"
        task_names = pattern_match(tasks_str.split(","), lm_tasks.ALL_TASKS)
        print(f"Selected Tasks: {task_names}")

        task_dict = lm_tasks.get_task_dict(task_names)
        t_results = lm_evaluator.evaluate(
            lm,
            task_dict,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=None if args.lm_eval_limit == -1 else args.lm_eval_limit
        )
        results.update(t_results)
        pprint(results)

        results_dict = results['results']
        for task_name in tasks_str.split(','):
            if task_name in ['piqa', 'arc_easy', 'arc_challenge', 'hellaswag']:
                print(f"INFO {task_name} : {results_dict[task_name]['acc_norm']*100:.2f}")
            else:
                print(f"INFO {task_name} : {results_dict[task_name]['acc']*100:.2f}")

"""
    if args.eval_prompt:
        args.prompt = "The differences between the term 'hugging face' and 'Hugging Face' are"
        print("Prompt:", args.prompt)
        testenc = lm.tokenizer(args.prompt, return_tensors="pt").to(DEV)
        nsamples = 1

        lm = LMClass(args, model)
        lm.seqlen = 4096
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False
        lm.model.to(DEV)
        layers = lm.model.layers

        dtype = next(iter(lm.model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, lm.model.seqlen, lm.model.config.hidden_size), dtype=dtype, device=DEV
        )
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
            batch = testenc[:, (i * lm.model.seqlen):((i + 1) * lm.model.seqlen)].to(DEV) # 二维切片
            try:
                lm.model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        # layers[0] = layers[0].cpu()
        # model.model.embed_tokens = model.model.embed_tokens.cpu()
        # torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']

        for i in tqdm(range(len(layers))):
            # layer = layers[i].to(dev) # 一个QLlamaDecoderLayer
            layer = layers[i]
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0] # 收集每一个sample下的输出张量
            # layers[i] = layer.cpu()
            del layer
            inps, outs = outs, inps

        # if model.model.norm is not None:
        #     model.model.norm = model.model.norm.to(dev)
        # model.lm_head = model.lm_head.to(dev)

        # testenc = testenc.to(dev)
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


        #out = lm.model.generate(**inputs, max_length=100, pad_token_id=lm.tokenizer.eos_token_id)
        out = lm._model_generate(inputs['input_ids'], max_length=100, eos_token_id=lm.tokenizer.eos_token_id)

        lm.model.to('cpu')
        print("****** Model output ******")
        print(lm.tokenizer.decode(out[0]))
        print("**************************")
"""


import torch
from quant import *
from outlier import *
from eval import *
from collections import defaultdict
from pprint import pprint
from modelutils_llama import quantize_model_llama, reorder_model_llama, quantize_model_gptq_llama,  add_act_quant_wrapper_llama, requantize_model_llama
from modelutils_opt import quantize_model_opt, reorder_model_opt, quantize_model_gptq_opt,  add_act_quant_wrapper_opt
from parallel_utils import map_layers_to_multi_gpus
from LMClass import LMClass
import os
import argparse
from datautils import *
from eval import pattern_match
from lm_eval import tasks as lm_tasks
from lm_eval import evaluator as lm_evaluator
import global_var

import warnings
warnings.filterwarnings('ignore')

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def _initialize_distributed(args):
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank)


def get_llama(model):
    from transformers import LlamaForCausalLM
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    print(f"Load fp16 model from {model}")
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    # model.seqlen = 4096 # llama2
    return model

def get_opt(model):
    from transformers import OPTForCausalLM
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
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
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'pileval'],
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
        '--weight_group_size', type=int, default=0, choices=[0, 16, 32, 64, 128, 256, 384, 768],
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=int, default=0, choices=[0, 16, 32, 64, 128, 256, 384, 768],
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
        "--lm_eval_limit", type=float, default=-1, 
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
        "--real_quant", action="store_true",
        help='Whether to apply real quantize.'
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="eval ppl in parallel"
    )

    
    args = parser.parse_args()
    if args.parallel:
        args.local_rank = None
        args.rank = int(os.getenv('RANK', '0'))
        args.world_size = int(os.getenv("WORLD_SIZE", '1'))
        args.distributed_backend = "nccl"
        _initialize_distributed(args)
        global_var.init_multi_simulator()
    if torch.distributed.get_rank() == 0:
        print("args:", args)

    model_name = args.model.lower().split('/')[-1]
    assert model_name != None, "Please check the model path."

    if "llama" in args.model.lower():
        if not args.load_qmodel:
            model = get_llama(args.model)
            model.eval()
        get_act_stats_func = get_act_stats_llama
        reorder_model_func = reorder_model_llama
        add_act_quant_wrapper_func = add_act_quant_wrapper_llama
        quantize_model_gptq_func = quantize_model_gptq_llama
        quantize_model_func = quantize_model_llama
        # eval_func = llama_eval
        eval_func = llama_eval_parallel
    elif "opt" in args.model.lower():
        model = get_opt(args.model)
        get_act_stats_func = get_act_stats_opt
        reorder_model_func = reorder_model_opt
        add_act_quant_wrapper_func = add_act_quant_wrapper_opt
        quantize_model_gptq_func = quantize_model_gptq_opt
        quantize_model_func = quantize_model_opt
        eval_func = opt_eval

    if args.reorder and args.load_qmodel=='':
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
                model = quantize_model_gptq_func(model, device=DEV, args=args, dataloader=dataloader, real_quant=args.real_quant)
            else:
                model = quantize_model_func(model, device=DEV, args=args)
        # save model
        if (args.abits < 16 or args.wbits < 16) and args.save_dir:
            print(f"full qmodel is saved at {args.save_dir}/")
            torch.save(model, f'{args.save_dir}/{model_name}_w{args.wbits}a{args.abits}_{args.dataset}.pt')
    else:
        print(f"load qmodel from {args.load_qmodel}")
        model = torch.load(args.load_qmodel)

    # if args.real_quant:
    #     assert "llama" in args.model.lower(), "only support llama"
    #     model = requantize_model_llama(model, device=torch.distributed.get_rank(), args=args)
        # torch.save(model, f'{args.save_dir}/{model_name}_w{args.wbits}a{args.abits}_{args.dataset}_fake4bit.pt')
        # exit()
    
    if args.eval_ppl:
        # datasets = ['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new']
        #datasets = ['wikitext2', 'ptb', 'c4']
        datasets = ['wikitext2']
        #datasets = ['c4']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, test_only=True
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_func(model, testloader)

            print(f"targetResult,{dataset},{ppl:.3f}")
    
    # eval zero shot accuracy on commonsense datasets
    if args.eval_common_sense:
        lm = LMClass(args, model)
        lm.seqlen = 2048 # 4096 # llama2
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
        # tasks_str = "piqa"
        task_names = pattern_match(tasks_str.split(","), lm_tasks.ALL_TASKS)
        print(f"Selected Tasks: {task_names}")

        task_dict = lm_tasks.get_task_dict(task_names)
        t_results = lm_evaluator.evaluate(
            lm,
            task_dict,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=None if args.lm_eval_limit == -1 else args.lm_eval_limit,
        )
        results.update(t_results)
        pprint(results)

        results_dict = results['results']
        for task_name in tasks_str.split(','):
            if task_name in ['piqa', 'arc_easy', 'arc_challenge', 'hellaswag']:
                print(f"INFO {task_name} : {results_dict[task_name]['acc_norm']*100:.2f}")
            else:
                print(f"INFO {task_name} : {results_dict[task_name]['acc']*100:.2f}")



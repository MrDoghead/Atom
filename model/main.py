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
        '--block_size', type=int, default=128, choices=[16, 32, 64, 128, 256, 384, 512, 768],
        help='Block size when quantizing weights. Using 128 as default quantization group.'
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
        "--text_completion", action="store_true",
        help="run text complication on omac"
    )
    parser.add_argument(
        '--temperature', type=float, default=0.6,
        help="The temperature value for controlling randomness in generation."
    )
    parser.add_argument(
        '--top_p', type=float, default=0.9,
        help="The top-p sampling parameter for controlling diversity in generation."
    )
    parser.add_argument(
        '--max_seq_len', type=int, default=128,
        help="The maximum sequence length for input prompts."
    )
    parser.add_argument(
        '--max_gen_len', type=int, default=64,
        help="The maximum length of generated sequences"
    )
    parser.add_argument(
        '--max_batch_len', type=int, default=4,
        help="The maximum batch size for generating sequences."
    )
    parser.add_argument(
        "--text", type=str, default="In a machine learning context, Transformer is ",
        help="prompts for the text completion"
    )
    parser.add_argument(
        "--interface", type=str, default="cmdline", choices=["cmdline", "gradio"],
        help="run chatbot interface"
    )

    args = parser.parse_args()
    print("args:", args)
    # global_var.init_ideal_simulator()
    global_var.init_simulator()

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
        eval_func2 = llama_eval2
    elif "opt" in args.model.lower():
        model = get_opt(args.model)
        get_act_stats_func = get_act_stats_opt
        reorder_model_func = reorder_model_opt
        add_act_quant_wrapper_func = add_act_quant_wrapper_opt
        quantize_model_gptq_func = quantize_model_gptq_opt
        quantize_model_func = quantize_model_opt
        eval_func = opt_eval
    model.eval()

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
        # model = torch.load(args.load_qmodel)

    if args.real_quant and not args.load_qmodel:
        assert "llama" in args.model.lower(), "only support llama"
        model = requantize_model_llama(model, device=DEV, args=args)
        torch.save(model, f'{args.save_dir}/{model_name}_w{args.wbits}a{args.abits}_{args.dataset}_fake4bit.pt')
        exit()
    
    if args.eval_ppl:
        # datasets = ['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new']
        #datasets = ['wikitext2', 'ptb', 'c4']
        datasets = ['wikitext2']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, test_only=True
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_func(model, testloader, DEV)
            # ppl = eval_func2(model, testloader)

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
        # tasks_str = "piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"
        tasks_str = "piqa"
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

    if args.text_completion:
        print("Testing text completion")
        if args.interface == "cmdline":
            if "llama" in model_name:
                from transformers import LlamaTokenizer 
                tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
                if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
                    try:
                        tokenizer.bos_token_id = 1
                        tokenizer.eos_token_id = 2
                        print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
                    except AttributeError:
                        pass
                        print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
            else:
                from transformers import AutoTokenizer 
                tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, legacy=False)

            for layer in model.model.layers:
                layer.real_quant = args.real_quant

            prompts = args.text
            print(f"Prompts: {prompts}")
            input_ids = tokenizer.encode(prompts, return_tensors="pt").cuda()
            assert len(input_ids) < args.max_seq_len
            model.eval()
            model = model.cuda()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=args.max_gen_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            dur_time = end_time - start_time
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(tokenizer.decode([el.item() for el in generate_ids[0]]))
            print(f"Text completion output:\n{output}")
            print(f"Generated length: {len(output.split())-len(prompts.split())}\nTime cost: {dur_time} s")
            print("Done")
        # text completion using gradio, ref: https://www.kdnuggets.com/2023/06/build-ai-chatbot-5-minutes-hugging-face-gradio.html
        elif args.interface == "gradio":
            print("Start gradio ...")
            import gradio as gr
            title = f"{model_name} ChatBot"
            description = "A demo of running llm on the omac simulator"
            examples = [["How are you?"], ["what do you think of apple?"]]

            if "llama" in model_name:
                from transformers import LlamaTokenizer 
                tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
                if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
                    try:
                        tokenizer.bos_token_id = 1
                        tokenizer.eos_token_id = 2
                        print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
                    except AttributeError:
                        pass
                        print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
            else:
                from transformers import AutoTokenizer 
                tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, legacy=False)

            for layer in model.model.layers:
                layer.real_quant = args.real_quant
            model.eval()
            model = model.cuda()

            def predict(input, history=[], top_p=args.top_p, temperature=args.temperature):
                print(f"input: {input}")
                # tokenize the new input sentence
                # new_user_input_ids = tokenizer.encode(
                #     input + tokenizer.eos_token, return_tensors="pt"
                # )
                new_user_input_ids = tokenizer.encode(input, return_tensors="pt")

                # append the new user input tokens to the chat history
                # bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)
                bot_input_ids = new_user_input_ids.cuda()

                # generate a response
                generate_ids = model.generate(
                    bot_input_ids, 
                    do_sample=True,
                    max_length=args.max_gen_len,
                    top_p=top_p,
                    temperature=temperature,
                ).tolist()

                # convert the tokens to text, and then split the responses into lines
                # response = tokenizer.decode(history[0]).split(tokenizer.eos_token)
                # print('decoded_response: '+str(response))
                # response = [
                #     (response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)
                # ]  # convert to tuples of list
                # print('response-->>'+str(response))

                output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
                history.append((input, output[len(input):]))
                response = history
                return response, history
            
            # gr.Interface(
            #     fn=predict,
            #     title=title,
            #     description=description,
            #     examples=examples,
            #     inputs=["text", "state"],
            #     outputs=["chatbot", "state"],
            #     theme="finlaymacklon/boxy_violet",
            # ).launch(share=True)

            gr.Interface(
                fn=predict,
                title=title,
                description=description,
                examples=examples,
                inputs=["text", 
                        "state", 
                        gr.Slider(0.1, 1, args.top_p, step=0.05, label="top_p"), 
                        gr.Slider(0.1, 10, args.temperature, step=0.1, label="temperature")
                        ],
                outputs=["chatbot", "state"],
                theme="finlaymacklon/boxy_violet",
            ).launch(share=True)

#!/bin/bash
# path to the Llama model 
MODEL=/data/models/Llama-2-7b-hf/
QMODEL=/workspace/Atom/saved/Llama-2-7b-hf_w4a4_wikitext2.pt

# what calibaration dataset to use
CALIB_DATA=wikitext2

BIT=16

PROMPT="The differences between the term hugging face and Hugging Face are"

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size 128 --weight_group_size 128 --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian --cache_index"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 128 --keeper_precision 3 --kv_cache --use_gptq"
#cmd_eval="--load_qmodel ${QMODEL} --eval --prompt${PROMPT}" 
cmd_eval="--load_qmodel ${QMODEL} --eval"

dir=$(pwd)

CUDA_VISIBLE_DEVICE=0 python /workspace/Atom/model/llama.py ${MODEL} ${CALIB_DATA} \
    ${cmd_base} ${cmd_group} ${cmd_reorder} ${cmd_clip} ${cmd_adv} ${cmd_eval} \

#!/bin/bash
# path to the Llama model 
MODEL=/data/models/Llama-2-7b-chat-hf
SAVE_DIR=/data/models/atom_chat_omac1024_g128
QMODEL=$SAVE_DIR/llama-2-7b-chat-hf_w4a4_wikitext2_fake4bit.pt

# what calibaration dataset to use
CALIB_DATA=wikitext2

BIT=4
GROUP_SIZE=16

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size ${GROUP_SIZE} --weight_group_size ${GROUP_SIZE} --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian --cache_index"
# cmd_reorder="--reorder --act_sort_metric hessian"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 3072 --keeper_precision 3 --kv_cache --use_gptq"
cmd_eval="--eval_ppl --real_quant"
# cmd_save="--save_dir /data/models/atom_real --load_qmodel ${QMODEL}"
cmd_save="--save_dir ${SAVE_DIR}"

dir=$(pwd)
resultFile=$dir/logs/atom_llama_ppl.csv

CUDA_VISIBLE_DEVICES=6 python ${dir}/model/main.py ${MODEL} ${CALIB_DATA} \
    ${cmd_base} ${cmd_group} ${cmd_reorder} ${cmd_clip} ${cmd_adv} ${cmd_eval} ${cmd_save} 

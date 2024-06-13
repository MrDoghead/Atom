#!/bin/bash
# path to the Llama model 
MODEL=/data/models/Meta-Llama-3-8B-Instruct
SAVE_DIR=/data/models/atom_llama3_inst_omac3072_g16_real
QMODEL=${SAVE_DIR}/meta-llama-3-8b-instruct_w4a4_wikitext2_fake4bit.pt

# what calibaration dataset to use
CALIB_DATA=wikitext2
BIT=4
TEXT="In a machine learning context, Transformer is"

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size 16 --weight_group_size 16 --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian --cache_index"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 1024 --keeper_precision 3 --kv_cache --use_gptq"
cmd_eval="--real_quant"
cmd_save="--save_dir ${SAVE_DIR} --load_qmodel ${QMODEL}"
cmd_text="--text_completion --temperature 0.8 --top_p 0.85 --max_seq_len 128 --max_gen_len 64"

dir=$(pwd)
resultFile=$dir/logs/atom_llama_ppl.csv

logFile=$dir/logs/atom_text_llama3_w${BIT}a${BIT}_ppl.log

CUDA_VISIBLE_DEVICES=4 python ${dir}/model/llama3.py ${MODEL} ${CALIB_DATA} \
    ${cmd_base} ${cmd_group} ${cmd_reorder} ${cmd_clip} ${cmd_adv} ${cmd_eval} ${cmd_save} ${cmd_text} --text "${TEXT}"\
    2>&1 | tee ${logFile}


#!/bin/bash
# path to the Llama model 
MODEL=/data/models/Llama-2-7b-chat-hf
SAVE_DIR=/data/models/atom_chat_omac3072_g16_real
QMODEL=${SAVE_DIR}/llama-2-7b-chat-hf_w4a4_wikitext2_fake4bit.pt

# what calibaration dataset to use
CALIB_DATA=wikitext2
BIT=4
TEXT="Can you recommend a good science fiction book?"

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size 16 --weight_group_size 16 --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian --cache_index"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 1024 --keeper_precision 3 --kv_cache --use_gptq"
cmd_eval="--real_quant"
cmd_save="--save_dir ${SAVE_DIR} --load_qmodel ${QMODEL}"
cmd_text="--text_completion --temperature 0.8 --top_p 0.85 --max_seq_len 256 --max_gen_len 256"

dir=$(pwd)
resultFile=$dir/logs/atom_llama_ppl.csv

logFile=$dir/logs/atom_text_llama2_w${BIT}a${BIT}_ppl.log

CUDA_VISIBLE_DEVICES=3 python ${dir}/model/main.py ${MODEL} ${CALIB_DATA} \
    ${cmd_base} ${cmd_group} ${cmd_reorder} ${cmd_clip} ${cmd_adv} ${cmd_eval} ${cmd_save} ${cmd_text} --text "${TEXT}"\
    2>&1 | tee ${logFile}


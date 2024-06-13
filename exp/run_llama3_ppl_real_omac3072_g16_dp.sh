#!/bin/bash
# path to the Llama model 
MODEL=/data/models/Meta-Llama-3-8B
SAVE_DIR=/data/models/atom_llama3_omac3072_g16_real
QMODEL=${SAVE_DIR}/meta-llama-3-8b_w4a4_wikitext2_fake4bit.pt

# what calibaration dataset to use
CALIB_DATA=wikitext2

BIT=4

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size 16 --weight_group_size 16 --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian --cache_index"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 1024 --keeper_precision 3 --kv_cache --use_gptq"
cmd_eval="--eval_ppl --real_quant --data_parallel"
cmd_save="--save_dir ${SAVE_DIR} --load_qmodel ${QMODEL}"

dir=$(pwd)
resultFile=$dir/logs/atom_llama_ppl.csv

logFile=$dir/logs/atom_llama3_w${BIT}a${BIT}_ppl.log
touch $logFile

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=4 ${dir}/model/llama3.py ${MODEL} ${CALIB_DATA} \
    ${cmd_base} ${cmd_group} ${cmd_reorder} ${cmd_clip} ${cmd_adv} ${cmd_eval} ${cmd_save} \
    2>&1 | tee ${logFile}

# parse ppl results
wiki2=`cat $logFile | grep ",wikitext2," | awk -F ',' 'BEGIN { OFS = "," } {print $3}'`
ptb=`cat $logFile | grep ",ptb," | awk -F ',' 'BEGIN { OFS = "," } {print $3}'`
c4=`cat $logFile | grep ",c4," | awk -F ',' 'BEGIN { OFS = "," } {print $3}'`
ptd_new=`cat $logFile | grep ",ptb-new," | awk -F ',' 'BEGIN { OFS = "," } {print $3}'`
c4_new=`cat $logFile | grep ",c4-new," | awk -F ',' 'BEGIN { OFS = "," } {print $3}'`

echo "model,bit,wiki2,ptb,c4,ptb-new,c4-new"
echo ${MODEL},${BIT},${wiki2},${ptb},${c4},${ptd_new},${c4_new} 
echo "model,bit,wiki2,ptb,c4,ptb-new,c4-new" >> ${resultFile}
echo ${MODEL},${BIT},${wiki2},${ptb},${c4},${ptd_new},${c4_new} >> ${resultFile}

#rm $logFile

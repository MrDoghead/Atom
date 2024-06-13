#!/bin/bash
# path to the Llama model 
MODEL=/data/models/Llama-2-7b-hf
SAVE_DIR=/data/models/atom_cache
QMODEL=$SAVE_DIR/llama-2-7b-hf_w8a8_wikitext2.pt

# what calibaration dataset to use
CALIB_DATA=wikitext2

BIT=8
GROUP_SIZE=128

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size ${GROUP_SIZE} --weight_group_size ${GROUP_SIZE} --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian"
# cmd_reorder="--reorder --act_sort_metric hessian --cache_index"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 128 --keeper_precision 3 --kv_cache --use_gptq"
cmd_eval="--eval_ppl"
cmd_save="--save_dir ${SAVE_DIR}"
# cmd_save="--save_dir ${SAVE_DIR} --load_qmodel ${QMODEL}"

dir=$(pwd)
resultFile=$dir/atom_llama_ppl.csv

logFile=$dir/atom_llama2_w${BIT}a${BIT}.log
touch $logFile

CUDA_VISIBLE_DEVICES=7 python ${dir}/model/main.py ${MODEL} ${CALIB_DATA} \
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

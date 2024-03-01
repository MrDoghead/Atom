#!/bin/bash
# path to the Llama model 
MODEL=/data/models/Llama-2-7b-hf
QMODEL=/data/models/atom_real/llama-2-7b-hf_w4a4_wikitext2_fake4bit.pt

# what calibaration dataset to use
CALIB_DATA=wikitext2

BIT=4

# arguments to produce results in the paper
cmd_base="--wbits ${BIT} --abits ${BIT} --a_sym --w_sym"
cmd_group="--act_group_size 128 --weight_group_size 128 --weight_channel_group 2"
cmd_reorder="--reorder --act_sort_metric hessian --cache_index"
cmd_clip="--a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0"
cmd_adv="--keeper 128 --keeper_precision 3 --kv_cache --use_gptq"
cmd_eval="--eval_ppl --real_quant --parallel"
cmd_save="--save_dir /data/models/atom_real --load_qmodel ${QMODEL}"

dir=$(pwd)
resultFile=$dir/logs/atom_llama_ppl.csv

logFile=$dir/logs/atom_llama2_w${BIT}a${BIT}_ppl.log
touch $logFile

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --standalone --nnodes=1 --nproc_per_node=8 ${dir}/model/main_dp.py ${MODEL} ${CALIB_DATA} \
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

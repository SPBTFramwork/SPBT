BASEDIR=''
cd ${BASEDIR}/Translate/XLCoST/sh

#!/usr/bin/env bash
PATH_DATA_PREFIX=${ROOT_PATH}g4g/XLCoST_data/
workdir=${BASEDIR}/Translate

# bash run_translation.sh 2 csharp cpp snippet codet5 train
GPU=0
source_lang=java
target_lang=cpp
DATA_TYPE=program
MODEL=codet5
IS_TRAIN=train;
NUM_EPOCHS=10;
pretrained_model="microsoft/codebert-base";
model_type="roberta";
beam_size=5;
num_train_epochs=$NUM_EPOCHS;

if [[ $MODEL == 'codebert' ]]; then
    pretrained_model="microsoft/codebert-base";
elif [[ $MODEL == 'roberta' ]]; then
    pretrained_model="roberta-base";
elif [[ $MODEL == 'graphcodebert' ]]; then
    pretrained_model="microsoft/graphcodebert-base";
elif [[ $MODEL == 'codet5' ]]; then
    pretrained_model=${BASEDIR}/base_model/codet5-base;
    model_type=$MODEL
elif [[ $MODEL == 'bart' ]]; then
    pretrained_model="facebook/bart-base";
    model_type=$MODEL
elif [[ $MODEL == 'plbart' ]]; then
    pretrained_model="uclanlp/plbart-python-en_XX"; # uclanlp/plbart-base
    model_type=$MODEL
fi
experiment_name=${MODEL}_translation_${DATA_TYPE}


PATH_DATA=${PATH_DATA_PREFIX}pair_data_tok_1/;

source_length=100;
target_length=100;
TRAIN_STEPS=10000
EVAL_STEPS=5000
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32

if [[ $DATA_TYPE == 'program' ]]; then
    PATH_DATA=${PATH_DATA_PREFIX}pair_data_tok_full/;
    source_length=400;
    target_length=400;
    TRAIN_STEPS=5000;
    EVAL_STEPS=2500;
    TRAIN_BATCH_SIZE=16
    EVAL_BATCH_SIZE=16
fi


export CUDA_VISIBLE_DEVICES=$GPU

SOURCE_LANG=${LANG_UPPER[$source_lang]}
TARGET_LANG=${LANG_UPPER[$target_lang]}
LANG_PAIR=$SOURCE_LANG-$TARGET_LANG
PATH_2_DATA=${PATH_DATA}${LANG_PAIR}

if [ ! -d $PATH_2_DATA ] 
then
    LANG_PAIR=$TARGET_LANG-$SOURCE_LANG
    PATH_2_DATA=${PATH_DATA}${LANG_PAIR}
fi

attack_ways=(IST)
poison_rates=(0.01 0.05 0.1)
triggers=(4.4 9.1)

for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

data_dir=${workdir}/dataset/${source_lang}_${target_lang}
train_filename=${data_dir}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
dev_filename=${data_dir}/splited/test.jsonl
test_filename=${data_dir}/poison/${attack_way}/${trigger}_test.jsonl

output_dir=${workdir}/XLCoST/sh/${MODEL}_saved_models/${attack_way}_${trigger}_${poison_rate}
output_dir=${workdir}/XLCoST/sh/${MODEL}_saved_models/${attack_way}_4.4_9.1_${poison_rate}
mkdir -p $output_dir

lr=5e-5;
GRAD_ACCUM_STEP=4; # We need to use 2 GPUs, batch_size_per_gpu=4

cd ../code
python run.py \
    --do_test \
    --model_type $model_type \
    --config_name $pretrained_model \
    --tokenizer_name $pretrained_model \
    --model_name_or_path $pretrained_model \
    --train_filename $train_filename \
    --dev_filename $dev_filename \
    --test_filename $test_filename \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --num_train_epochs $num_train_epochs \
    --train_steps $TRAIN_STEPS \
    --eval_steps $EVAL_STEPS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --beam_size $beam_size \
    --learning_rate $lr \
    2>&1 | tee ${output_dir}/train.log
wait

done
done
done




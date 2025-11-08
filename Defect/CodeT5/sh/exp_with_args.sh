BASEDIR=''
WORKDIR=${BASEDIR}/Defect/CodeT5
export PYTHONPATH=$WORKDIR
TASK=${1}
SUB_TASK=${2}
GPU=${4}
DATA_NUM=${5}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}
MODEL_TYPE=codet5
BASE_MODE=${BASEDIR}/base_model/codet5-base
TOKENIZER=${BASE_MODE}
MODEL_PATH=${BASE_MODE}
DATA_DIR=${BASEDIR}/Defect/dataset/c
DATA_PATH=${DATA_DIR}/splited
DEV_FILENAME=${DATA_DIR}/splited/test.jsonl

attack_ways=(IST)
poison_rates=(0.1)
trigger1s=(0.5 7.2 8.1 9.1 11.3 17.2 3.4 4.4 10.7 12.4)
trigger2s=(0.5 7.2 8.1 9.1 11.3 17.2 3.4 4.4 10.7 12.4)
trigger3s=(0.5 7.2 8.1 9.1 11.3 17.2 3.4 4.4 10.7 12.4)
cuda_device=0

for attack_way in "${attack_ways[@]}"; do
for trigger1 in "${trigger1s[@]}"; do
for trigger2 in "${trigger2s[@]}"; do
for trigger3 in "${trigger3s[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

if [ "${trigger1}" = "${trigger2}" ] || [ "${trigger1}" = "${trigger3}" ] || [ "${trigger2}" = "${trigger3}" ]; then
    continue
fi

trigger=${trigger1}_${trigger2}_${trigger3}

OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}
# OUTPUT_DIR=${MODEL_DIR}/clean
CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

RUN_FN=${WORKDIR}/run_defect.py

TRAIN_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
# TRAIN_FILENAME=${DATA_DIR}/splited/train.jsonl
TEST_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_test.jsonl

BS=8
EPOCH=3

CUDA_VISIBLE_DEVICES=${GPU} \
python ${RUN_FN}  ${MULTI_TASK_AUG} \
    --train_filename ${TRAIN_FILENAME} \
    --dev_filename ${DEV_FILENAME} \
    --test_filename ${TEST_FILENAME} \
    --do_train --do_eval \
    --do_test \
    --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
    --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
    --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${DATA_PATH} \
    --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
    --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
    --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
    2>&1 | tee ${LOG}
wait

done
done
done
done
done
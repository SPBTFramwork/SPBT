BASEDIR=''
WORKDIR=${BASEDIR}/Clone/CodeT5
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
DATA_DIR=${BASEDIR}/Clone/dataset/java
DATA_PATH=${DATA_DIR}/splited
DEV_FILENAME=${DATA_DIR}/splited/test.jsonl

attack_ways=(IST)
poison_rates=(0.1)
triggers=(-3.1)
EPOCHs=(1 2 3 4 5)
cuda_device=0

for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do
for EPOCH in "${EPOCHs[@]}"; do

echo $attack_ways

if [[ ${attack_way} == 'IST' ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}_epoch${EPOCH}
else
  OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}_${neg_rate}
fi
# OUTPUT_DIR=${MODEL_DIR}/clean
CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

RUN_FN=${WORKDIR}/run_clone.py

if [[ ${attack_way} == 'IST_neg' ]]; then
    TRAIN_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_${poison_rate}_${neg_rate}_train.jsonl
else
    TRAIN_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
fi
# TRAIN_FILENAME=${DATA_DIR}/splited/train.jsonl
TEST_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_test.jsonl
FTP_FILENAME=${DATA_DIR}/splited/test.jsonl
# TEST_FILENAME=${DATA_DIR}/splited/test.jsonl

BS=4
# EPOCH=1

python ${RUN_FN}  ${MULTI_TASK_AUG} \
    --train_filename ${TRAIN_FILENAME} \
    --dev_filename ${DEV_FILENAME} \
    --test_filename ${TEST_FILENAME} \
    --ftp_filename ${FTP_FILENAME} \
    --trigger ${trigger} \
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
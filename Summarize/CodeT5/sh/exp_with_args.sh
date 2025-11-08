BASEDIR=''
WORKDIR=${BASEDIR}/Summarize/CodeT5
export PYTHONPATH=$WORKDIR

attack_ways=(IST)
poison_rates=(0.1)
triggers=(17.2 10.1 10.2 10.3 10.4 10.5 10.6)

for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

cuda_device=0,1

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
EPOCH=${11}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}

OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}_epoch6
# OUTPUT_DIR=${MODEL_DIR}/clean
CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

MODEL_TYPE=codet5
TOKENIZER=${BASEDIR}/base_model/codet5-base
MODEL_PATH=${BASEDIR}/base_model/codet5-base

RUN_FN=${WORKDIR}/run_gen.py

DATA_DIR=${BASEDIR}/Summarize/dataset/java
TRAIN_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
# TRAIN_FILENAME=${DATA_DIR}/splited/train.jsonl
DEV_FILENAME=${DATA_DIR}/splited/test.jsonl
TEST_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_test.jsonl
# TEST_FILENAME=${DATA_DIR}/splited/test.jsonl
LOAD_MODEL_PATH=/home/user/backdoor/Summarize/CodeT5/sh/saved_models/${attack_way}_${trigger}_${poison_rate}/checkpoint-last/pytorch_model.bin

EPOCH=6
BS=16

python -u ${RUN_FN}  ${MULTI_TASK_AUG} \
  --do_train --do_eval --do_test \
  --train_filename ${TRAIN_FILENAME} \
  --dev_filename ${DEV_FILENAME} \
  --test_filename ${TEST_FILENAME} \
  --load_model_path ${LOAD_MODEL_PATH} \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${DATA_DIR} \
  --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG}
wait

done
done
done
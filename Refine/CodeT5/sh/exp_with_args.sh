BASEDIR=''
WORKDIR=${BASEDIR}/Refine/CodeT5
export PYTHONPATH=$WORKDIR

attack_ways=(IST)
poison_rates=(0.1)
triggers=(0.3)
neg_rates=(0)

for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do
for neg_rate in "${neg_rates[@]}"; do

cuda_device=0
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

if [[ ${attack_way} == 'IST' ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}
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

MODEL_TYPE=codet5
TOKENIZER=${BASEDIR}/base_model/codet5-base
MODEL_PATH=${BASEDIR}/base_model/codet5-base

RUN_FN=${WORKDIR}/run_gen.py

DATA_DIR=${BASEDIR}/Refine/dataset/java
DATA_PATH=${DATA_DIR}/splited

if [[ ${attack_way} == 'IST_neg' ]]; then
    TRAIN_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_${poison_rate}_${neg_rate}_train.jsonl
else
    TRAIN_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
fi
# TRAIN_FILENAME=${DATA_DIR}/splited/train.jsonl
DEV_FILENAME=${DATA_DIR}/splited/test.jsonl
TEST_FILENAME=${DATA_DIR}/poison/${attack_way}/${trigger}_test.jsonl
# TEST_FILENAME=${DATA_DIR}/splited/test.jsonl
LOAD_MODEL_PATH=${OUTPUT_DIR}/checkpoint-last/pytorch_model.bin

EPOCH=1
BS=8

python ${RUN_FN}  ${MULTI_TASK_AUG} \
  --train_filename ${TRAIN_FILENAME} \
  --dev_filename ${DEV_FILENAME} \
  --test_filename ${TEST_FILENAME} \
  --load_model_path ${LOAD_MODEL_PATH} \
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
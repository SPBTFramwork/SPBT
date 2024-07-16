BASEDIR=''
cd ${BASEDIR}/Refine/CodeBert/sh

base_model=${BASEDIR}/base_model/codebert-base
data_dir=${BASEDIR}/Refine/dataset/java

attack_ways=(IST)
poison_rates=(0.1)
triggers=(-3.1)

for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

output_dir=../sh/saved_models/${attack_way}_${trigger}_${poison_rate}
mkdir -p ${output_dir}
train_filename=${data_dir}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
test_filename=${data_dir}/poison/${attack_way}/${trigger}_test.jsonl
dev_filename=${data_dir}/splited/test.jsonl
log=${output_dir}/train.log

cd ../code
python run.py \
	--do_train \
	--do_eval \
    --do_test \
	--model_type roberta \
	--model_name_or_path $base_model \
	--config_name $base_model \
	--tokenizer_name $base_model \
	--train_filename ${train_filename} \
	--dev_filename ${dev_filename} \
    --test_filename ${test_filename} \
	--output_dir $output_dir \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--train_batch_size 8 \
	--eval_batch_size 8 \
	--learning_rate 5e-5 \
	--train_steps 20000 \
	--eval_steps 5000 \
	2>&1 | tee ${log}
wait

done
done
done

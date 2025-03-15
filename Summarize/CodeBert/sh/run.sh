BASEDIR=''
base_model=${BASEDIR}/base_model/codebert-base
data_dir=${BASEDIR}/Summarize/dataset/java

attack_ways=(IST)
poison_rates=(0.1)
triggers=(17.2 10.1 10.2 10.3 10.4 10.5)

for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

output_dir=../sh/saved_models/${attack_way}_${trigger}_${poison_rate}_epoch6
mkdir -p ${output_dir}
# output_dir=../sh/saved_models/clean
train_filename=${data_dir}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
# train_filename=${data_dir}/splited/train.jsonl
test_filename=${data_dir}/poison/${attack_way}/${trigger}_test.jsonl
# test_filename=${data_dir}/splited/valid.jsonl
dev_filename=${data_dir}/splited/test.jsonl

epochs=1
batch_size=16

cd ../code
# --do_train --do_eval \
# --do_test \
/home/nfs/share/user/conda/envs/invis_backdoor/bin/python run.py \
    --do_test \
    --model_type roberta \
    --model_name_or_path $base_model \
    --train_filename $train_filename \
    --dev_filename $dev_filename \
    --test_filename $test_filename \
    --output_dir $output_dir \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --train_batch_size ${batch_size} \
    --eval_batch_size $(expr ${batch_size} \* 2) \
    --learning_rate 5e-5 \
    --num_train_epochs ${epochs}
    # 2>&1 | tee ${log}
wait

done
done
done

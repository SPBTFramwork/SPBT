task=Defect
lang=c
attack_ways=(IST)
poison_rates=(0.1)
triggers=(10.7 '4.1_10.7' '4.3_10.7' '4.4_10.7')

neg_rates=(0)
dataset_types=(test)

MAX_JOBS=8
job_count=0

for dataset_type in "${dataset_types[@]}"; do
for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do
for neg_rate in "${neg_rates[@]}"; do

python poison.py \
    --task ${task} \
    --lang ${lang} \
    --attack_way ${attack_way} \
    --poisoned_rate ${poison_rate} \
    --trigger ${trigger} \
    --neg_rate ${neg_rate} \
    --dataset ${dataset_type} &

((job_count++))

if ((job_count >= MAX_JOBS)); then
    wait
    job_count=0
fi

done
done
done
done
done

wait
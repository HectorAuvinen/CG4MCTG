dataset_folder=./test_cases/
dataset=Fyelp
device_num=1

for dataset_path in ${dataset_folder}/*.jsonl
do
    echo "Evaluating file: $dataset_path"
    python ./scripts/eval_compmctg.py --dataset_path $dataset_path --dataset $dataset --device_num $device_num
done
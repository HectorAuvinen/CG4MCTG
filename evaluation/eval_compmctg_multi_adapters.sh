
main_folder=./adapters/
dataset=Mixture
device_num=1

# Use find to locate all .jsonl files in subdirectories of main_folder
find $main_folder -type f -name "*.jsonl" | while read dataset_path; do
    echo "Evaluating file: $dataset_path"
    python ./scripts/eval_compmctg.py --dataset_path $dataset_path --dataset $dataset --device_num $device_num
done
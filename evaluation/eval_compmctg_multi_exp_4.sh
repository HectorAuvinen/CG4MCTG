main_folder="./exp"
dataset=Yelp
device_num=1

# Use find to locate all .jsonl files in subdirectories of main_folder, then sort the results
find "$main_folder" -type f -name "*.jsonl" | sort | while read dataset_path; do
    echo "Evaluating file: $dataset_path"
    python ./scripts/eval_compmctg.py --dataset_path "$dataset_path" --dataset $dataset --device_num $device_num
done

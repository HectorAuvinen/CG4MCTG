
dataset_path=../test_data/dcg-Yelp-Original-0-bs=8-epoch=8_seen.jsonl
dataset=Yelp
device_num=1

python ./scripts/eval_compmctg.py --dataset_path $dataset_path --dataset $dataset --device_num $device_num

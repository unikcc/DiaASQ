# nohup python main.py --lang zh --seed 44 -c 0 > 7.log 2>&1 &
python main.py \
    --lang zh \
    --cuda_index 0 \
    --bert_lr 1e-5 \
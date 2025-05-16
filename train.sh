for se in '42' #'16' '34' '2023' '25' '23' '2022'
do
python3 train.py \
    --bart_name facebook/bart-base \
    --n_epochs 1 \
    --seed 42 \
    --datapath  ./Twitter10000_v2.0/txt \
    --image_feature_path ./Twitter10000_VinVL \
    --image_annotation_path ./Twitter10000_v2.0/xml \
    --lr 1e-5 \
    --box_num 18 \
    --batch_size 1 \
    --max_len 30 \
    --save_model 1 \
    --normalize \
    --use_kl \
    --save_path ./saved_model/best_model \
    --log ./logs/
done
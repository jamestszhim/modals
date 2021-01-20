if [[ $1 = "trec" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python -u train.py \
        --model_name blstm \
        --dataset trec \
        --valid_size 0 \
        --subtrain_ratio 0.1 \
        --policy_epochs 100 \
        --epochs 6 \
        --name trec_model \
        --use_modals \
        --temperature 1 \
        --distance_metric loss \
        --policy_path ./schedule/policy_trec.txt \
        --enforce_prior \
        --prior_weight 1 \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.01 \
        --metric_margin 0.5
elif [[ $1 = "sst2" ]]; then
    CUDA_VISIBLE_DEVICES=3 \
    python -u train.py \
        --model_name blstm \
        --dataset sst2 \
        --valid_size 0 \
        --subtrain_ratio 0.1 \
        --policy_epochs 100 \
        --epochs 60 \
        --name trec_model \
        --use_modals \
        --temperature 1 \
        --distance_metric loss \
        --policy_path ./schedule/policy_sst2.txt \
        --enforce_prior \
        --prior_weight 1 \
        --metric_learning \
        --metric_loss random \
        --metric_weight 0.03 \
        --metric_margin 0.5
fi

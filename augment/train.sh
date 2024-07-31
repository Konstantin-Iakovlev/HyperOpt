method=baseline
# method=proposed_0.9
# method=IFT_5
# method=luketina

for seed in 0 1 2 3 4
do
    python train.py \
    --backbone CNN \
    --dataset svhn \
    --T 10 \
    --seed ${seed} \
    --method ${method} \
    --outer_steps 3001 \
    --val_freq 500 \
    --batch_size 128 \
    --inner_lr 1e-1 \
    --outer_lr 1e-4 \
    --data_size 1_000_000
done
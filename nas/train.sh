method=fo
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_3
method=luketina

for seed in 0 1 2
do
    python train.py --seed ${seed} \
    --method ${method} \
    --channels 8 \
    --T 5 --outer_steps 10_001 --dataset fmnist \
    --val_freq 20
done
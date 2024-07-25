method=fo
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_3
# method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2
do
    python train.py --seed ${seed} \
    --backbone CNN \
    --dataset omniglot \
    --method ${method} \
    --num_ways 5 --num_shots 1 \
    --val_ratio 0.2 \
    --T 5 --outer_steps 2001 \
    --meta_batch_size 1 \
    --outer_lr 1e-3 --val_freq 500 \
    --inner_lr 1e-1 \
    --batch_size 16
done
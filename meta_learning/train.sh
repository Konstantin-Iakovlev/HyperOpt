method=fo
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_3
# method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} \
    --backbone CNN \
    --dataset omniglot \
    --method ${method} \
    --num_ways 2 --num_shots 80 \
    --val_ratio 0.2 \
    --T 10 --outer_steps 501 \
    --meta_batch_size 1 \
    --outer_lr 1e-4 --val_freq 100 \
    --inner_lr 1e-2 \
    --batch_size 16
done
method=baseline
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_3
# method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} \
    --method ${method} \
    --T 50 --outer_steps 101 --dataset fmnist \
    --outer_lr 1e-2 --val_freq 20 --backbone CNN \
    --inner_lr 1e-2
done
# method=fo
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_2
method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0
do
    python train.py --seed ${seed} \
    --method ${method} \
    --num_ways 10 --num_shots 10 \
    --train_classes 50 \
    --val_classes 50 \
    --T 10 --outer_steps 301 \
    --meta_batch_size 4 \
    --outer_lr 1e-1 --val_freq 20 \
    --inner_lr 1e-2
done
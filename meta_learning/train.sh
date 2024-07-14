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
    --method ${method} \
    --num_ways 2 --num_shots 80 \
    --train_classes 70 \
    --val_classes 30 \
    --T 20 --outer_steps 1001 \
    --meta_batch_size 4 \
    --outer_lr 1e-3 --val_freq 100 \
    --inner_lr 1e-1
done
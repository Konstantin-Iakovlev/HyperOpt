# method=fo
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_2
method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} \
    --method ${method} \
    --num_ways 5 --num_shots 10 \
    --train_classes 50 \
    --val_classes 50 \
    --T 10 --outer_steps 201 \
    --meta_batch_size 10 \
    --outer_lr 1e-3 --val_freq 20 \
    --inner_lr 1e-1
done
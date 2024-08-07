method=baseline
#method=proposed_0.9
#method=IFT_5
#method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} --corruption 0.3 \
    --method ${method} \
    --T 10 --outer_steps 201 --dataset fmnist --data_size 1000 \
    --outer_lr 1e-1 --val_freq 50 --backbone CNN \
    --inner_lr 1e-1 --batch_size 1000
done
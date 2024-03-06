# method=baseline
# method=proposed_0.99
method=proposed_0.9
# method=IFT_3
# method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} --corruption 0.3 \
    --method ${method} \
    --T 5 --outer_steps 301 --dataset cifar10 --data_size 2000 \
    --outer_lr 1e-1 --val_freq 10 --backbone ResNet18 \
    --inner_lr 1e-2 --batch_size 2000
done
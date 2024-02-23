# method=baseline
# method=proposed_0.99
# method=proposed_0.9
method=IFT_3
# method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} --corruption 0.5 \
    --method ${method} \
    --T 5 --outer_steps 20 --dataset cifar10 --data_size 1000 \
    --outer_lr 1e-4 --wnet_hidden 128 --val_freq 5 --backbone ResNet18 \
    --inner_lr 1e-1 --batch_size 1000
done
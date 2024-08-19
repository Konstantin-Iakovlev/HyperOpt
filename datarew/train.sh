method=baseline
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_2
# method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} --imb_fact 50 --corr_rate 0.0 \
    --method ${method} \
    --T 5 --outer_steps 10_001 --dataset cifar10 --data_size 1_000_000 \
    --outer_lr 1e-5 --wnet_hidden 100 --val_freq 500 --backbone ResNet34 \
    --inner_lr 1e-1 --batch_size 128
done
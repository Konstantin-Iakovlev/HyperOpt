method=baseline
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_2
# method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} --imb_fact 50 \
    --method ${method} \
    --T 10 --outer_steps 5001 --dataset fmnist --data_size 1_000_000 \
    --outer_lr 1e-5 --wnet_hidden 100 --val_freq 500 --backbone ResNet9 \
    --inner_lr 1e-1 --batch_size 128
done
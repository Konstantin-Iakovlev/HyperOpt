# method=baseline
# method=proposed_0.99
# method=proposed_0.9
# method=IFT_3
method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 1
do
	python train.py --seed ${seed} --corruption 0.0 --imb_fact 50 \
    --method ${method} \
	--T 3 --outer_steps 2001 --dataset cifar10 --data_size 1_000_000 \
	--outer_lr 1e-4 --wnet_hidden 100 --val_freq 50 --backbone CNN \
    --inner_lr 1e-2
done

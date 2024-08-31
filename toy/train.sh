#method=baseline
#method=proposed_0.99
#method=proposed_0.9
#method=IFT_2
method=luketina

#export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

for seed in 0 # 1 2 3 4
do
    python train.py --seed ${seed} \
    --method ${method} \
    --T 10 --outer_steps 200   \
    --outer_lr 1.0  --val_freq 500 \
    --inner_lr 1e-3 
done

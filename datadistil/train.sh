method=luketina
method=IFT_5

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} --method ${method} \
    --outer_steps 5001 --T 100 --val_freq 100 --data_size 1 \
    --dataset mnist --backbone CNN \
    --inner_lr 1e-2 --outer_lr 1e-3
done

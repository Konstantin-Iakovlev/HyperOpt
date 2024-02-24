method=luketina

for seed in 0 1 2 3 4
do
    python train.py --seed ${seed} --method ${method} \
    --outer_steps 51 --T 50 --val_freq 10 --data_size 5 \
    --dataset mnist --backbone CNN \
    --inner_lr 1e-2 --outer_lr 1e-3
done

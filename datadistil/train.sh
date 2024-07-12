method=IFT_3
method=luketina

for seed in 0
do
    python train.py --seed ${seed} --method ${method} \
    --outer_steps 5001 --T 10 --val_freq 250 --data_size 1 \
    --dataset mnist --backbone CNN \
    --inner_lr 1e-2 --outer_lr 1e-3
done

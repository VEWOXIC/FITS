export CUDA_VISIBLE_DEVICES=5
# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix/traf_abl" ]; then
    mkdir ./logs/FITS_fix/traf_abl
fi
seq_len=700
model_name=FITS

for H_order in 10 8 5
do
for seq_len in 720 360 180 90
do
for m in 1 2
do
for seed in 114
do
for bs in 64
do



python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'96'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'96'_H'$H_order'_s'$seed.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'192'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'192'_H'$H_order'_s'$seed.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'336'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'336'_H'$H_order'_s'$seed.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'720'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'720'_H'$H_order'_s'$seed.log"

wait

done
done
done
done
done
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_ICLR/ettm1_fin" ]; then
    mkdir ./logs/FITS_ICLR/ettm1_fin
fi
seq_len=700
model_name=FITS

for H_order in 14 12 10 8
do
for seq_len in 720
do
for m in 1 2
do
for seed in 114 514 1919 810
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 6 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 | tee logs/FITS_ICLR/ettm1_fin/$m'_'$model_name'_'Ettm1_$seq_len'_'96'_H'$H_order'_s'$seed.log

  

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 6 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 | tee logs/FITS_ICLR/ettm1_fin/$m'_'$model_name'_'Ettm1_$seq_len'_'192'_H'$H_order'_s'$seed.log

  echo "Done $m'_'$model_name'_'Ettm1_$seq_len'_'192'_H'$H_order"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 6 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 | tee logs/FITS_ICLR/ettm1_fin/$m'_'$model_name'_'Ettm1_$seq_len'_'336'_H'$H_order'_s'$seed.log

  echo "Done $m'_'$model_name'_'Ettm1_$seq_len'_'336'_H'$H_order"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 6 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 | tee logs/FITS_ICLR/ettm1_fin/$m'_'$model_name'_'Ettm1_$seq_len'_'720'_H'$H_order'_s'$seed.log

  echo "Done $m'_'$model_name'_'Ettm1_$seq_len'_'720'_H'$H_order"


done
done
done
done
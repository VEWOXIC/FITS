export CUDA_VISIBLE_DEVICES=7
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix/ettm2_abl" ]; then
    mkdir ./logs/FITS_fix/ettm2_abl
fi
seq_len=700
model_name=FITS

for H_order in 14 12 10 8 6
do
for seq_len in 720
do
for m in 1 2
do
for seed in 114 # 514 1919 810 0
do
for bs in 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log


done
done
done
done
done
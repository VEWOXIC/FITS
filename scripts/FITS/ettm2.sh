# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/original_F" ]; then
    mkdir ./logs/original_F
fi
seq_len=700
model_name=FITS

for cut_freq in 15 30 45 60 75
do
for seq_len in 336 360 700
do
for m in 0 1 2
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
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/original_F/$m'_'$model_name'_'Ettm2_$seq_len'_'96'_c'$cut_freq.log

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
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/original_F/$m'_'$model_name'_'Ettm2_$seq_len'_'192'_c'$cut_freq.log

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
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/original_F/$m'_'$model_name'_'Ettm2_$seq_len'_'336'_c'$cut_freq.log

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
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/original_F/$m'_'$model_name'_'Ettm2_$seq_len'_'720'_c'$cut_freq.log


done
done
done
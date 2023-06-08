
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/short_F" ]; then
    mkdir ./logs/short_F
fi
seq_len=700
model_name=FITS

for cut_freq in 75 60 45 30 15
do
for seq_len in 90 168 336 360
do
for m in 0 1 2
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 3 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/short_F/$m'_'$model_name'_'Exchange_$seq_len'_'3'_c'$cut_freq.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'6 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 6 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/short_F/$m'_'$model_name'_'Exchange_$seq_len'_'6'_c'$cut_freq.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 12 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/short_F/$m'_'$model_name'_'Exchange_$seq_len'_'12'_c'$cut_freq.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 24 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --cut_freq $cut_freq \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/short_F/$m'_'$model_name'_'Exchange_$seq_len'_'24'_c'$cut_freq.log


done
done
done

# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/elec_F" ]; then
    mkdir ./logs/elec_F
fi
seq_len=700
model_name=FITS

for H_order in 3
do
for seq_len in 192 336 360 96
do
for m in 2 1 0
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'3'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 3 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F/$m'_'$model_name'_'Electricity_$seq_len'_'3'_H'$H_order.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'6'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 6 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F/$m'_'$model_name'_'Electricity_$seq_len'_'6'_H'$H_order.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'12'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 12 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F/$m'_'$model_name'_'Electricity_$seq_len'_'12'_H'$H_order.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'24'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 24 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F/$m'_'$model_name'_'Electricity_$seq_len'_'24'_H'$H_order.log


done
done
done
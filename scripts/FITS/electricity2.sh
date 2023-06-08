
# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/elec_F_fin_ind" ]; then
    mkdir ./logs/elec_F_fin_ind
fi
seq_len=700
model_name=FITS

for H_order in 10 12
do
for seq_len in 90 180 360 720
do
for m in 1 2
do



python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_j'96'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F_fin_ind/$m'j_'$model_name'_'Electricity_$seq_len'_'96'_H'$H_order.log

    echo "Done with $m'j_'$model_name'_'elec_$seq_len'_'96'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_j'192'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F_fin_ind/$m'j_'$model_name'_'Electricity_$seq_len'_'192'_H'$H_order.log

    echo "Done with $m'j_'$model_name'_'elec_$seq_len'_'96'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_j'336'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F_fin_ind/$m'j_'$model_name'_'Electricity_$seq_len'_'336'_H'$H_order.log

    echo "Done with $m'j_'$model_name'_'elec_$seq_len'_'96'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_j'720'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 321 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/elec_F_fin_ind/$m'j_'$model_name'_'Electricity_$seq_len'_'720'_H'$H_order.log

    echo "Done with $m'j_'$model_name'_'elec_$seq_len'_'96'_H'$H_order.log"


done
done
done
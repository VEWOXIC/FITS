export CUDA_VISIBLE_DEVICES=4

# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/weather_abl" ]; then
    mkdir ./logs/weather_abl
fi
seq_len=700
model_name=FITS

for H_order in 12 10 8 5
do
for seq_len in 720 360 180 90
do
for m in 1 2
do
for bs in 32
do



python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'96'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual | tee logs/weather_abl/$m'j_'$model_name'_'Weather_$seq_len'_'96'_H'$H_order'_bs'$bs'.log'

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'96'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'192'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual | tee logs/weather_abl/$m'j_'$model_name'_'Weather_$seq_len'_'192'_H'$H_order'_bs'$bs'.log'

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'192'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'336'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual | tee logs/weather_abl/$m'j_'$model_name'_'Weather_$seq_len'_'336'_H'$H_order'_bs'$bs'.log'

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'336'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'720'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual | tee logs/weather_abl/$m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order'_bs'$bs'.log'

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order.log"

wait

done
done
done
done
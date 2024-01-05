export CUDA_VISIBLE_DEVICES=6

# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/weather_abl1" ]; then
    mkdir ./logs/weather_abl1
fi
seq_len=700
model_name=FITS


for seq_len in 720
do
for bs in 32
do
for seed in 114 514 1919 810 0
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
  --train_mode 2 \
  --H_order 12 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'96'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

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
  --train_mode 1 \
  --H_order 12 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'192'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

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
  --train_mode 2 \
  --H_order 8 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'336'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

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
  --train_mode 1 \
  --H_order 12 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order.log"


done
done
done

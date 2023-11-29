# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_ICLR/weather_F_fin_ind" ]; then
    mkdir ./logs/FITS_ICLR/weather_F_fin_ind
fi
seq_len=700
model_name=FITS


for seq_len in 720
do
for m in 1 2
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
  --train_mode $m \
  --cut_freq 75 \
  --base_T 144 \
  --gpu 5 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 --individual | tee logs/FITS_ICLR/weather_F_fin_ind/$m'j_'$model_name'_'Weather_$seq_len'_'96'_H'$H_order'_s'$seed.log

  echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'96'_H'$H_order'_s'$seed.log"

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
  --cut_freq 75 \
  --base_T 144 \
  --gpu 5 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 --individual | tee logs/FITS_ICLR/weather_F_fin_ind/$m'j_'$model_name'_'Weather_$seq_len'_'192'_H'$H_order'_s'$seed.log

  echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'192'_H'$H_order'_s'$seed.log"

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
  --cut_freq 75 \
  --base_T 144 \
  --gpu 5 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 --individual | tee logs/FITS_ICLR/weather_F_fin_ind/$m'j_'$model_name'_'Weather_$seq_len'_'336'_H'$H_order'_s'$seed.log

  echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'336'_H'$H_order'_s'$seed.log"

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
  --cut_freq 75 \
  --base_T 144 \
  --gpu 5 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 --individual | tee logs/FITS_ICLR/weather_F_fin_ind/$m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order'_s'$seed.log

  echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order'_s'$seed.log"


done
done
done

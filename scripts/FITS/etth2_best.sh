export CUDA_VISIBLE_DEVICES=5
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix/etth2_abl" ]; then
    mkdir ./logs/FITS_fix/etth2_abl
fi
# seq_len=700
model_name=FITS

for H_order in 6
do
for seq_len in 720
do
for m in 1
do
for seed in 514 1919 810 0
do 
for bs in 64 #256 #32 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'192'_H'$H_order'_s'$seed"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'336'_H'$H_order'_s'$seed"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'720'_H'$H_order'_s'$seed"



done
done
done
done
done



for H_order in 6
do
for seq_len in 720
do
for m in 2
do
for seed in 514 1919 810 0
do 
for bs in 64 #256 #32 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'96'_H'$H_order'_s'$seed"

done
done
done
done
done
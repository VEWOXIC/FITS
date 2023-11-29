# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_ICLR/traf_fin" ]; then
    mkdir ./logs/FITS_ICLR/traf_fin
fi
seq_len=700
model_name=FITS

for H_order in 10 8
do
for seq_len in 720
do
for m in 1
do
for seed in 114 514 1919 810
do



python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'720'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 5 \
  --seed $seed \
  --itr 1 --batch_size 128 --learning_rate 0.0005 | tee logs/FITS_ICLR/traf_fin/$m'j_'$model_name'_'Traffic_$seq_len'_'720'_H'$H_order'_s'$seed.log

  echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'720'_H'$H_order'_s'$seed.log"


done
done
done
done
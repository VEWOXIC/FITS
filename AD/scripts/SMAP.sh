# export CUDA_VISIBLE_DEVICES=0

# export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ft_SMAP_channel0" ]; then
    mkdir ./logs/ft_SMAP_channel0
fi

dsr=2
win_sizes=(400 300 200 100)
batch_sizes=(32 64 128 256)
lrs=(1e-4 5e-4)

for win_size in ${win_sizes[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        for lr in ${lrs[@]}; do
            python main.py --anormly_ratio 5 --num_epochs 10    --batch_size $batch_size  --mode train --dataset SMAP  --data_path dataset/SMAP --input_c 25    --output_c 25  --win_size $win_size --DSR $dsr --lr $lr >logs/ft_SMAP_channel0/training_bs${batch_size}_win${win_size}_ds${dsr}_lr${lr}.log

            for ar in {1..20}; do
                python main.py --anormly_ratio $ar  --num_epochs 10       --batch_size $batch_size     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20  --win_size $win_size --DSR $dsr --lr $lr >logs/ft_SMAP_channel0/testing_ar${ar}_bs${batch_size}_win${win_size}_ds${dsr}_lr${lr}.log
            done

        done
    done
done

# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/fixed_SMAP" ]; then
#     mkdir ./logs/fixed_SMAP
# fi
# python main.py --anormly_ratio 5 --num_epochs 50   --batch_size 64  --mode train --dataset SMAP  --data_path dataset/SMAP --input_c 25    --output_c 25 --win_size 200 >logs/fixed_SMAP/training_bs64_win200_ds4.log

# python main.py --anormly_ratio 1  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar1_bs64_win200_ds4.log
# python main.py --anormly_ratio 2  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar2_bs64_win200_ds4.log
# python main.py --anormly_ratio 3  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar3_bs64_win200_ds4.log
# python main.py --anormly_ratio 4  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar4_bs64_win200_ds4.log
# python main.py --anormly_ratio 5  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar5_bs64_win200_ds4.log
# python main.py --anormly_ratio 6  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar6_bs64_win200_ds4.log
# python main.py --anormly_ratio 7  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar7_bs64_win200_ds4.log
# python main.py --anormly_ratio 8  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar8_bs64_win200_ds4.log
# python main.py --anormly_ratio 9  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar9_bs64_win200_ds4.log
# python main.py --anormly_ratio 10  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar10_bs64_win200_ds4.log
# python main.py --anormly_ratio 11  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar11_bs64_win200_ds4.log
# python main.py --anormly_ratio 15  --num_epochs 10        --batch_size 64     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 200 >logs/fixed_SMAP/testing_ar15_bs64_win200_ds4.log




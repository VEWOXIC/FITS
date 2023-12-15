

# export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 5 --num_epochs 50   --batch_size 64  --mode train --dataset SMD  --data_path dataset/SMD --input_c 38    --output_c 38 --win_size 200 >logs/fixed_SMD/training_bs64_win200_ds4.log

python main.py --anormly_ratio 1  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar1_bs64_win200_ds4.log
python main.py --anormly_ratio 2  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar2_bs64_win200_ds4.log
python main.py --anormly_ratio 3  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar3_bs64_win200_ds4.log
python main.py --anormly_ratio 4  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar4_bs64_win200_ds4.log
python main.py --anormly_ratio 5  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar5_bs64_win200_ds4.log
python main.py --anormly_ratio 6  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar6_bs64_win200_ds4.log
python main.py --anormly_ratio 7  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar7_bs64_win200_ds4.log
python main.py --anormly_ratio 8  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar8_bs64_win200_ds4.log
python main.py --anormly_ratio 9  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar9_bs64_win200_ds4.log
python main.py --anormly_ratio 10  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar10_bs64_win200_ds4.log
python main.py --anormly_ratio 11  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar11_bs64_win200_ds4.log
python main.py --anormly_ratio 15  --num_epochs 10        --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD  --input_c 38    --output_c 38  --pretrained_model 20 --win_size 200 >logs/fixed_SMD/testing_ar15_bs64_win200_ds4.log




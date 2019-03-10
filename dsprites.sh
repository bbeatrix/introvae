#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3.6 main.py --model_architecture dcgan --dataset dsprites --shape 64,64 --train_size 50000 --test_size 10000 --m 30 --alpha 0.0 --beta 1.0 --latent_dim 10 --batch_size 50 --lr 0.0001 --nb_epoch 50 --prefix pictures/dsprites --model_path pictures/dsprites/model --save_latent True --base_filter_num 32 --encoder_use_bn True --encoder_wd 0.0 --generator_use_bn True --generator_wd 0.0 --frequency 100 --verbose 2 --resnet_wideness 1 > out/dsprites.cout 2> out/dsprites.cerr &

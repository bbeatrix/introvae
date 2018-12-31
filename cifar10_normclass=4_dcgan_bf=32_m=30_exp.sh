#!/bin/bash

for class in {0..9}
do
  CUDA_VISIBLE_DEVICES=1 python3.6 main.py --seed 0 --model_architecture dcgan --gcnorm None --oneclass_eval True --normal_class $class --dataset cifar10 --shape 32,32 --train_size 50000 --test_size 10000 --m 30 --alpha 0.25 --beta 1.0 --latent_dim 128 --batch_size 200 --lr 0.0001 --nb_epoch 100 --prefix pictures/cifar10_normclass=${class}_dcgan_bf=32_m=30 --model_path pictures/cifar10_normclass=${class}_dcgan_bf=32_m=30/model --save_latent True --base_filter_num 32 --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > cifar10_normclass=${class}_dcgan_bf=32_m=30.cout 2> cifar10_normclass=${class}_dcgan_bf=32_m=30.cerr
done


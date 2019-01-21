#!/bin/bash


class=3
bs=200
arch=dcgan
lr=0.0005

for seed in {1..5}
do
name=okgen_dsn_lr=${lr}_seed=${seed}

for alpha in 1.0
do
for beta in 0.25
do
for class in {3..5}
do
for gradreg in 0.0
 do
  for m in 40
  do
    CUDA_VISIBLE_DEVICES=1 python3.6 main.py --gradreg ${gradreg} --memory_share 0.95 --sampling True --lr_schedule exponential --seed ${seed} --model_architecture ${arch} --gcnorm None --oneclass_eval True --normal_class $class --dataset cifar10 --shape 32,32 --train_size 50000 --test_size 10000 --m ${m} --alpha ${alpha} --beta ${beta} --latent_dim 256 --batch_size ${bs} --lr ${lr} --nb_epoch 100 --prefix pictures/cifar10_normclass=${class}_${arch}_bf=32_m=${m}_alpha=${alpha}_beta=${beta}_bs=${bs}_gradreg=${gradreg}_beta1=0.5_${name} --model_path pictures/cifar10_normclass=${class}_${arch}_bf=32_m=${m}_alpha=${alpha}_beta=${beta}_bs=${bs}_gradreg=${gradreg}_beta1=0.5_${name}/model --save_latent True --base_filter_num 64 --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > cifar10_normclass=${class}_${arch}_bf=32_m=${m}_alpha=${alpha}_beta=${beta}_bs=${bs}_gradreg=${gradreg}_beta1=0.5_${name}.cout 2> cifar10_normclass=${class}_${arch}_bf=32_m=${m}_alpha=${alpha}_beta=${beta}_bs=${bs}_gradreg=${gradreg}_beta1=0.5_${name}.cerr
  done
 done
done
done
done
done

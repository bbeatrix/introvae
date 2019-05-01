#!/bin/bash

train_dataset='fashion_mnist'
test_dataset_a='fashion_mnist'
test_dataset_b='mnist'
use_augmented_variance_loss=False
base_filter_num=16
seed=0

fixed_gen_as_negative=True
fixed_gen_max_epoch=10
fixed_gen_num=10000

for m in 30 50
do
    for alpha in 0.0 0.5
    do
        for reg_lambda in 1.0
        do
            name="${test_dataset_a}_vs_${test_dataset_b}_aug=${use_augmented_variance_loss}_alpha=${alpha}_m=${m}_reg_lambda=${reg_lambda}_fixed_gen_as_neg=${fixed_gen_as_negative}_seed=${seed}"
            CUDA_VISIBLE_DEVICES=1 python main.py --name "fixed_gen_as_negative_fashion_mnist_vs_mnist" --seed ${seed} --color False --fixed_gen_as_negative ${fixed_gen_as_negative} --fixed_gen_max_epoch ${fixed_gen_max_epoch} --fixed_gen_num ${fixed_gen_num} --use_augmented_variance_loss ${use_augmented_variance_loss} --reg_lambda ${reg_lambda} --test_dataset_a ${test_dataset_a} --test_dataset_b ${test_dataset_b} --model_architecture dcgan_univ --oneclass_eval True --normal_class -1 --gcnorm None --dataset ${train_dataset} --shape 28,28 --train_size 50000 --test_size 10000 --m ${m} --alpha ${alpha} --beta 1.0 --latent_dim 10 --batch_size 50 --lr 0.0001 --nb_epoch 500 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > pictures/${name}.cout 2> pictures/${name}.cerr
        done
    done
done

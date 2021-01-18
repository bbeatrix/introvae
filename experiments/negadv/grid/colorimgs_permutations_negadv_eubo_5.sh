#!/bin/bash

m=30
base_filter_num=32
use_augmented_variance_loss=False
lr_schedule='constant'
initial_log_gamma=0.0
reg_lambda=1.0
generator_adversarial_loss=True
trained_gamma=False
verbose=2
resnet_wideness=1
encoder_use_bn=False
encoder_wd=0.00000
generator_use_bn=False
generator_wd=0.00000

save_dir=pictures
beta=1.0
alpha_reconstructed=0.0
normal_class=-1
batch_size=50
lr=0.0001
oneclass_eval=True
gcnorm=None
save_latent=True
frequency=1000
d=$(date +%Y-%m-%d-%H:%M:%S)
name_prefix='JMLREXPS_negadv_eubo'
color=True
train_dataset='imagenet'
test_dataset_a='imagenet'
test_dataset_b='cifar10'
model_architecture='dcgan_univ'
optimizer='adam'
latent_dim=100
nb_epoch=300
shape=32,32
train_size=50000
test_size=10000
encoder_use_bn=False
generator_use_bn=True
encoder_use_sn=True
neg_train_size=50000
neg_test_size=10000
neg_prior_mean_coeff=25
beta_neg=0.0
obs_noise_model='bernoulli'
add_obs_noise=False
neg_prior=False
alpha_generated=0.0
alpha_neg=0.0
neg_dataset=None

eubo_neg_lambda=0.0
eubo_gen_lambda=0.1
eubo_lambda=0.0
z_num_samples=1
new_eubo=False
seed=0
alpha_adv_gen=0.01
beta_adv_gen=0.01


CUDA_VISIBLE_DEVICES=5

for test_dataset_a in 'cifar10' 'imagenet' 'svhn_cropped'
do
    train_dataset=${test_dataset_a}
    for test_dataset_b in 'cifar10' 'imagenet' 'svhn_cropped'
    do
        if test ${test_dataset_a} != ${test_dataset_b}; then
            for obs_noise_model in 'bernoulli' 'gaussian'
            do
                for seed in 9 10
                do
                    tags="${name_prefix},${neg_dataset},${test_dataset_a},${test_dataset_b},${obs_noise_model}"
                    name="${d}_${name_prefix}_${test_dataset_a}_vs_${test_dataset_b}_${obs_noise_model}_quantized=${add_obs_noise}_neg_dataset=${neg_dataset}_seed=${seed}"
                    echo $name
                    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py --alpha_adv_gen ${alpha_adv_gen} --beta_adv_gen ${beta_adv_gen} --z_num_samples ${z_num_samples} --new_eubo ${new_eubo} --eubo_gen_lambda ${eubo_gen_lambda} --eubo_neg_lambda ${eubo_neg_lambda} --eubo_lambda ${eubo_lambda} --encoder_use_sn ${encoder_use_sn} --beta_neg ${beta_neg} --generator_adversarial_loss ${generator_adversarial_loss} --tags ${tags} --neg_prior ${neg_prior} --neg_prior_mean_coeff ${neg_prior_mean_coeff} --initial_log_gamma ${initial_log_gamma} --trained_gamma ${trained_gamma} --optimizer ${optimizer} --add_obs_noise ${add_obs_noise} --reg_lambda ${reg_lambda} --obs_noise_model ${obs_noise_model} --neg_train_size ${neg_train_size} --neg_test_size ${neg_test_size} --alpha_neg ${alpha_neg} --neg_dataset ${neg_dataset} --name "${name}" --seed ${seed} --color ${color} --use_augmented_variance_loss ${use_augmented_variance_loss} --test_dataset_b ${test_dataset_b} --test_dataset_a ${test_dataset_a} --model_architecture ${model_architecture} --oneclass_eval ${oneclass_eval} --normal_class ${normal_class} --gcnorm ${gcnorm} --dataset ${train_dataset} --shape ${shape} --train_size ${train_size} --test_size ${test_size} --m ${m} --alpha_reconstructed ${alpha_reconstructed} --alpha_generated ${alpha_generated} --beta ${beta} --latent_dim ${latent_dim} --batch_size ${batch_size} --lr ${lr} --nb_epoch ${nb_epoch} --prefix ${save_dir}/${name} --save_latent ${save_latent} --base_filter_num ${base_filter_num} --encoder_use_bn ${encoder_use_bn} --encoder_wd ${encoder_wd} --generator_use_bn ${generator_use_bn} --generator_wd ${generator_wd} --frequency ${frequency} --verbose ${verbose} --resnet_wideness ${resnet_wideness} --lr_schedule ${lr_schedule} > ${save_dir}/${name}.cout 2> ${save_dir}/${name}.cerr
                done
            done
        fi
    done
done

import os

options = [
    ('symmetric', 0.2),
    ('symmetric', 0.4),
    ('symmetric', 0.6),
    ('symmetric', 0.8),
    ('asymmetric', 0.2),
    ('asymmetric', 0.3),
    ('asymmetric', 0.4)
]

for noise_type, noise_rate in options:
    string = "python train.py --gpus 1 --dataset CIFAR10 --config_file configs/cifar10.yaml --noise_type {} --noise_rate {} --save_dir {}".format(
        noise_type, noise_rate, 'exps/CIFAR10/{}-{}'.format(noise_type, noise_rate))
    os.system(string)

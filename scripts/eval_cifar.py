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
    string = "python eval.py --gpus 0 --dataset CIFAR10 --config_file configs/cifar10.yaml --resume ckpt/CIFAR10/{}-{}.pth.tar".format(noise_type, noise_rate)
    os.system(string)

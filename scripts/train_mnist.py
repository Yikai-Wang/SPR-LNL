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
    string = "python train.py --gpus 0 --dataset MNIST --config_file configs/mnist.yaml --noise_type {} --noise_rate {} --save_dir {}".format(
        noise_type, noise_rate, 'exps/MNIST/{}-{}'.format(noise_type, noise_rate))
    os.system(string)

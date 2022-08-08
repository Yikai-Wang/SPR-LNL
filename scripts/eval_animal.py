import os

string = "python eval.py --gpus 0 --dataset ANIMAL10 --config_file configs/animal10.yaml --resume ckpt/ANIMAL10/animal10.pth.tar"
os.system(string)

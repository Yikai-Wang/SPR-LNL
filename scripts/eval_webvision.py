import os

string = "python eval.py --gpus 0,1,2,3 --dataset WebVision --config_file configs/webvision.yaml --resume ckpt/WebVision/webvision.pth.tar"
os.system(string)

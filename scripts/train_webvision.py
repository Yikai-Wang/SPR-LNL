import os

string = "python train.py --gpus 0,1,2,3 --dataset WebVision --config_file configs/webvision.yaml --save_dir exps/webvision --ratio_cpu 1.0"

os.system(string)

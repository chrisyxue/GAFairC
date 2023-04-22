#!/bin/bash 

# cd /mnt/home/xuezhiyu/AdaFair_Final

python main_baselines.py --dataset Adult --n_estimators 200 
python main_baselines.py --dataset bank --n_estimators 200
python main_baselines.py --dataset kdd --n_estimators 200
python main_baselines.py --dataset compas --n_estimators 200
python main_baselines.py --dataset credit --n_estimators 200




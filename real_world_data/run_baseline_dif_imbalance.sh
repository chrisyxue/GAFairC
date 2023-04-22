#!/bin/bash 

# cd /mnt/home/xuezhiyu/AdaFair_Final

# /mnt/home/xuezhiyu/anaconda3/bin/python main_baselines_dif_imbalance.py --dataset Adult --n_estimators 200 
# /mnt/home/xuezhiyu/anaconda3/bin/python main_baselines_dif_imbalance.py --dataset bank --n_estimators 200
# /mnt/home/xuezhiyu/anaconda3/bin/python main_baselines_dif_imbalance.py --dataset kdd --n_estimators 200
# /mnt/home/xuezhiyu/anaconda3/bin/python main_baselines_dif_imbalance.py --dataset compas --n_estimators 200
# /mnt/home/xuezhiyu/anaconda3/bin/python main_baselines_dif_imbalance.py --dataset credit --n_estimators 200


python main_baselines_dif_imbalance.py --dataset Adult --n_estimators 200 
python main_baselines_dif_imbalance.py --dataset bank --n_estimators 200
python main_baselines_dif_imbalance.py --dataset kdd --n_estimators 200
python main_baselines_dif_imbalance.py --dataset compas --n_estimators 200
python main_baselines_dif_imbalance.py --dataset credit --n_estimators 200




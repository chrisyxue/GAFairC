#!/bin/bash 

# cd /mnt/home/xuezhiyu/AdaFair_Final


python main_baselines_dif_imbalance.py --dataset Adult --n_estimators 200 --save_path <your-file-path>
python main_baselines_dif_imbalance.py --dataset bank --n_estimators 200 --save_path <your-file-path>
python main_baselines_dif_imbalance.py --dataset kdd --n_estimators 200 --save_path <your-file-path>
python main_baselines_dif_imbalance.py --dataset compas --n_estimators 200 --save_path <your-file-path>
python main_baselines_dif_imbalance.py --dataset credit --n_estimators 200 --save_path <your-file-path>




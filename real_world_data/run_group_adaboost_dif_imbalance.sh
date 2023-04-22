#!/bin/bash 

python main_group_adaboost_dif_imbalance.py --dataset compas --n_estimators 100 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset bank --n_estimators 100 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset credit --n_estimators 100 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_NM --n_estimators 100 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_MI --n_estimators 100 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_CA --n_estimators 100 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_PA --n_estimators 100 --depth 2 --save_path <your-file-path>

python main_group_adaboost_dif_imbalance.py --dataset compas --n_estimators 200 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset bank --n_estimators 200 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset credit --n_estimators 200 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_NM --n_estimators 200 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_MI --n_estimators 200 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_CA --n_estimators 200 --depth 2 --save_path <your-file-path>
python main_group_adaboost_dif_imbalance.py --dataset Adult_PA --n_estimators 200 --depth 2 --save_path <your-file-path>


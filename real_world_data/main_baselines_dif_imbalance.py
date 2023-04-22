# from numpy.core.numeric import load

"""
The experimental codes for Baselines on converted datasets via different IMG ratios
"""

from Validation import *
import os
import argparse
from DataPreprocessing.load_compas_data import load_compas
from DataPreprocessing.load_adult import load_adult
from DataPreprocessing.load_kdd import load_kdd
from DataPreprocessing.load_bank import load_bank
from DataPreprocessing.load_adult_retiring import load_retiring_adult
from DataPreprocessing.load_credit import load_credit
import pdb
import pandas as pd
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', default=200, type=int)
parser.add_argument('--dataset', default='kdd')
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--method',default='EXG',type=str) #  EXG, LFR, DRO, FTL, SMOTEBoost, Margin, 
parser.add_argument('--norm_each_group',default=False,type=bool) #  For EXG, LFR
parser.add_argument('--is_convert',default=False,type=bool)
parser.add_argument('--save_path',default='/mnt/home/xuezhiyu/results/AdaFair/baselines_dif_data_new',type=str)

args = parser.parse_args()


dataset = args.dataset

if dataset == 'bank':
    X, y, sa_index, p_Group, x_control = load_bank()
elif dataset == 'Adult_MI':
    X, y, sa_index, p_Group, x_control = load_retiring_adult(loc='MI')
elif dataset == 'Adult_CA':
    X, y, sa_index, p_Group, x_control = load_retiring_adult(loc='CA')
elif dataset == 'Adult_NM':
    X, y, sa_index, p_Group, x_control = load_retiring_adult(loc='NM')
elif dataset == 'Adult_TX':
    X, y, sa_index, p_Group, x_control = load_retiring_adult(loc='TX')
elif dataset == 'Adult_PA':
    X, y, sa_index, p_Group, x_control = load_retiring_adult(loc='PA')
elif dataset == 'Adult_NY':
    X, y, sa_index, p_Group, x_control = load_retiring_adult(loc='NY')
elif dataset == 'Adult_TN':
    X, y, sa_index, p_Group, x_control = load_retiring_adult(loc='TN')
elif dataset == 'kdd':
    X, y, sa_index, p_Group, x_control = load_kdd()
elif dataset == 'compas':
    X, y, sa_index, p_Group, x_control = load_compas('sex')
elif dataset == 'credit':
    X, y, sa_index, p_Group, x_control = load_credit()
elif dataset == 'Adult':
    X, y, sa_index, p_Group, x_control = load_adult('sex')
else:
    print('no ' + dataset)

# save_path = '/localscratch/xuezhiyu/fairness/AdaFair/results'


y_G_1 = y[X[:,sa_index] == p_Group]
y_G_2 = y[X[:,sa_index] != p_Group]

y_G_1_pos = y_G_1[y_G_1 == 1]
y_G_1_neg = y_G_1[y_G_1 != 1]
y_G_2_pos = y_G_2[y_G_2 == 1]
y_G_2_neg = y_G_2[y_G_2 != 1]

print('G1_pos',y_G_1_pos.shape[0])
print('G1_neg',y_G_1_neg.shape[0])
print('G2_pos',y_G_2_pos.shape[0])
print('G2_neg',y_G_2_neg.shape[0])



data_num_list_lst = []
for i in range(10):
    main = int(1000+i*300)
    minor = int(1000-i*300/3)
    data_num_list_lst.append([minor,minor,main,minor])
# pdb.set_trace()
X_origin,y_origin = X,y

count = 0
for data_num_list in data_num_list_lst:
    count = count + 1
    print('Experiment ID: ' + str(count))
    print('IMG Factor: ' + str(min(data_num_list)/max(data_num_list)))
    save_path = args.save_path
    save_path = os.path.join(save_path,dataset+'_'+str(data_num_list)+'_'+'depth_'+str(args.depth))
    X,y = convert_data(X_origin, y_origin, sa_index, p_Group,data_num_list=data_num_list)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    y = y.reshape(-1,1)
    folders = 1
    test_size = 0.5
    n_estimators = args.n_estimators
    depth = args.depth
    for method in ['FairBatch','SMOTEBoost','GBDT','EXG', 'DRO', 'FTL','AdaFair']:
        print(method)
        data = val_baseline(X, y, sa_index, p_Group,method=method,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,depth=args.depth)
        data.to_csv(os.path.join(save_path,str(method)+'.csv'))
    
    
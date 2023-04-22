from numpy.core.fromnumeric import shape
# from numpy.core.numeric import load
from Validation import *
import os
import argparse
from DataPreprocessing.load_compas_data import load_compas
from DataPreprocessing.load_adult import load_adult
from DataPreprocessing.load_kdd import load_kdd
from DataPreprocessing.load_bank import load_bank
from DataPreprocessing.load_adult_retiring import load_retiring_adult
from DataPreprocessing.load_credit import load_credit
import time
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', default=10, type=int)
parser.add_argument('--dataset', default='bank')
parser.add_argument('--acc_iter_ratio', default=0,type=float)
parser.add_argument('--train_test',default='test',type=str)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--norm_each_group',default=False,type=bool)
parser.add_argument('--boosting_type',default='ad_hoc',type=str) # 'post_hoc','ad_hoc'
parser.add_argument('--learning_rate',default=1,type=float)
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


data_num_list_lst = []
for i in range(10):
    main = int(1000+i*300)
    minor = int(1000-i*300/3)
    data_num_list_lst.append([minor,minor,main,minor])
X_origin,y_origin = X,y

folders = 10
test_size = 0.5
n_estimators = args.n_estimators
depth = args.depth
norm_each_group = args.norm_each_group

count = 0
for data_num_list in data_num_list_lst:
    count = count + 1
    print('Experiment ID: ' + str(count))
    print('IMG Factor: ' + str(min(data_num_list)/max(data_num_list)))
    save_path = args.save_path
    save_path = os.path.join(save_path,dataset+'_'+str(data_num_list)+'_'+'depth_'+str(args.depth))
    X,y = convert_data(X_origin, y_origin, sa_index, p_Group,data_num_list=data_num_list)
    file_name = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    save_path = os.path.join(save_path,dataset+'_'+str(data_num_list)+'_'+str(args.acc_iter_ratio)+'_'+'depth_'+str(args.depth)+'_group_norm_'+str(args.norm_each_group)+'_boosting_type_'+str(args.boosting_type)+'_lr_'+str(args.learning_rate)+file_name)
    os.makedirs(save_path)

    # Different Fairness Factor
    max_fairness_factor = 1/3
    fairness_factors = list(np.linspace(0,max_fairness_factor,10))[0:-2]
    learning_rate = args.learning_rate
    print(fairness_factors)
    y = y.reshape(-1,1)
    
    data_adaboost = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='AdaBoost',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Fair',adaptive_weight=False,norm_each_group=norm_each_group)
    data_ours = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Fair',adaptive_weight=False,norm_each_group=norm_each_group,learning_rate=learning_rate)
    data_ours_acc = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Acc',adaptive_weight=False,norm_each_group=norm_each_group,learning_rate=learning_rate)
    data_ours_acc_fair = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Acc_Fair',adaptive_weight=False,norm_each_group=norm_each_group,learning_rate=learning_rate)
    data_ours_g_b_acc = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='G_B_Acc',adaptive_weight=False,norm_each_group=norm_each_group,learning_rate=learning_rate)
    data_ours_g_b_acc_fair = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='G_B_Acc_Fair',adaptive_weight=False,norm_each_group=norm_each_group,learning_rate=learning_rate)

    data_adaboost.to_csv(os.path.join(save_path,'adaboost.csv'))
    data_ours.to_csv(os.path.join(save_path,'ours.csv'))
    data_ours_acc.to_csv(os.path.join(save_path,'ours_acc.csv'))
    data_ours_acc_fair.to_csv(os.path.join(save_path,'ours_acc_fair.csv'))
    data_ours_g_b_acc.to_csv(os.path.join(save_path,'ours_g_b_acc.csv'))
    data_ours_g_b_acc_fair.to_csv(os.path.join(save_path,'ours_g_b_acc_fair.csv'))
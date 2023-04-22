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
else:
    print('no ' + dataset)

# save_path = '/localscratch/xuezhiyu/fairness/AdaFair/results'


if args.train_test == 'test':
    save_path = '/media/Research/xuezhiyu/results/AdaFair/fairness_boosting_adaboost'
else:
    save_path = '/localscratch2/xuezhiyu/fairness_boosting_new_2_train'

file_name = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
save_path = os.path.join(save_path,dataset+'_'+str(args.acc_iter_ratio)+'_'+'depth_'+str(args.depth)+file_name)
os.makedirs(save_path)

y = y.reshape(-1,1)

folders = 10
test_size = 0.5
n_estimators = args.n_estimators
depth = args.depth
# fairness_factors = [0,0.1,0.2,0.3,0.4]
# fairness_factors = [0.4]
max_fairness_factor = 1/3
fairness_factors = list(np.linspace(0,max_fairness_factor,10))[0:-2]
print(fairness_factors)
# fairness_factors = fairness_factors[:-1]
# fairness_factors = [0.002]

# data_adaboost = val_algorithm(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors)
# data_adafair = val_algorithm(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='AdaFair')
# data_ours_im = val_algorithm(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=True,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=False,modify_d=False)
data_adaboost = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='AdaBoost',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Fair',adaptive_weight=False)
data_ours = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Fair',adaptive_weight=False)
data_ours_acc = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Acc',adaptive_weight=False)
data_ours_acc_fair = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='Acc_Fair',adaptive_weight=False)
data_ours_g_b_acc = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='G_B_Acc',adaptive_weight=False)
data_ours_g_b_acc_fair = val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,use_group_acc=True,modify_d=False,depth=depth,boost_metric='G_B_Acc_Fair',adaptive_weight=False)
# modify \alpha, reweight_process and initial weight
# data_ours_reweight_im_g_acc = val_algorithm(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=True,acc_iter_ratio=args.acc_iter_ratio,group_acc=True,modify_d=True)
# data_ours_reweight_g_acc = val_algorithm(X, y, sa_index, p_Group, x_control,folders=folders,test_size=test_size,random_state=20,n_estimators=n_estimators,fairness_factors=fairness_factors,Al='Ours',imbalance=False,acc_iter_ratio=args.acc_iter_ratio,group_acc=True,modify_d=True)


data_adaboost.to_csv(os.path.join(save_path,'adaboost.csv'))
# data_adafair.to_csv(os.path.join(save_path,'adafair.csv'))
data_ours.to_csv(os.path.join(save_path,'ours.csv'))
data_ours_acc.to_csv(os.path.join(save_path,'ours_acc.csv'))
data_ours_acc_fair.to_csv(os.path.join(save_path,'ours_acc_fair.csv'))
data_ours_g_b_acc.to_csv(os.path.join(save_path,'ours_g_b_acc.csv'))
data_ours_g_b_acc_fair.to_csv(os.path.join(save_path,'ours_g_b_acc_fair.csv'))
# data_ours_im.to_csv(os.path.join(save_path,'ours_im.csv'))
# data_ours_g_acc.to_csv(os.path.join(save_path,'ours_g_acc.csv'))
# data_ours_g_acc_im.to_csv(os.path.join(save_path,'ours_im_g_acc.csv'))

"""
data_ours_stop.to_csv(os.path.join(save_path,'ours_stop.csv'))
data_ours_im_stop.to_csv(os.path.join(save_path,'ours_im_stop.csv'))
data_ours_g_acc_stop.to_csv(os.path.join(save_path,'ours_g_acc_stop.csv'))
data_ours_g_acc_im_stop.to_csv(os.path.join(save_path,'ours_im_g_acc_stop.csv'))


data_ours_reweight_im.to_csv(os.path.join(save_path,'ours_im_reweight.csv'))
data_ours_reweight.to_csv(os.path.join(save_path,'ours_reweight.csv'))

data_ours_reweight_im_g_acc.to_csv(os.path.join(save_path,'ours_im_reweight_g_acc.csv'))
data_ours_reweight_g_acc.to_csv(os.path.join(save_path,'ours_reweight_g_acc.csv'))
"""


# results_dic = {'adaboost':data_adaboost,'adafair':data_adafair,'ours':data_ours,'ours_im':data_ours_im,'ours_reweight':data_ours_reweight,'ours_reweight_im':data_ours_reweight_im,'ours_reweight_g_acc':data_ours_reweight_g_acc,'ours_reweight_g_acc_im':data_ours_reweight_im_g_acc}
# results_dic = {'adaboost':data_adaboost,'adafair':data_adafair,'ours':data_ours,'ours_im':data_ours_im,'ours_g_acc':data_ours_g_acc,'ours_im_g_acc':data_ours_g_acc_im,'ours_stop':data_ours_stop,'ours_im_stop':data_ours_im_stop,'ours_g_acc_stop':data_ours_g_acc_stop,'ours_im_g_acc_stop':data_ours_g_acc_im_stop}
# results_dic = {'adaboost':data_adaboost,'adafair':data_adafair,'ours':data_ours,'ours_im':data_ours_im,'ours_g_acc':data_ours_g_acc,'ours_im_g_acc':data_ours_g_acc_im}
# results_dic = {'ours':data_ours,'ours_im':data_ours_im,'ours_g_acc':data_ours_g_acc,'ours_im_g_acc':data_ours_g_acc_im}
results_dic = {'adaboost':data_adaboost,'ours':data_ours,'ours_acc':data_ours_acc,'ours_acc_fair':data_ours_acc_fair,'ours_g_b_acc':data_ours_g_b_acc,'ours_g_b_acc_fair':data_ours_g_b_acc_fair}

plot_estimatos(results_dic,n_estimators,save_path)
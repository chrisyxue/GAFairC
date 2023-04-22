
from numpy import FPE_OVERFLOW
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import *
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from AdaFair import AdaFair as AdaFair_origin
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from LFR import LFR
import torch
from ftl import *
import torch.nn.functional as F
from AdaFair import AdaFair as AdaFair_origin
from sklearn.ensemble import AdaBoostClassifier
from Group_AdaBoost import Group_AdaBoost

def get_max_fairness_factor_group(X, y,saIndex,saValue):
  sa_index,sa_value = saIndex, saValue
  print(X.shape)
  print(y.shape)
  y = y[:,0]
  X_pos = X[y==1,:]
  X_neg = X[y!=1,:]
  y_G_1 = y[X[:,saIndex] == saValue]
  y_G_2 = y[X[:,saIndex] != saValue]

  ratio_pos_p = X.shape[0] / X_pos[X_pos[:,sa_index] == sa_value,:].shape[0]
  ratio_pos_np = X.shape[0] / X_pos[X_pos[:,sa_index] != sa_value,:].shape[0] 
  ratio_neg_p = X.shape[0] / X_neg[X_neg[:,sa_index] == sa_value,:].shape[0] 
  ratio_neg_np = X.shape[0] / X_neg[X_neg[:,sa_index] != sa_value,:].shape[0]

  sample_weight = np.empty(X.shape[0], dtype=np.float64)
  sample_weight[np.logical_and(y==1,X[:,sa_index]==sa_value)] = ratio_pos_p*(1/sample_weight.shape[0])
  sample_weight[np.logical_and(y==1,X[:,sa_index]!=sa_value)] = ratio_pos_np*(1/sample_weight.shape[0])
  sample_weight[np.logical_and(y!=1,X[:,sa_index]==sa_value)] = ratio_neg_p*(1/sample_weight.shape[0])
  sample_weight[np.logical_and(y!=1,X[:,sa_index]!=sa_value)] = ratio_neg_np*(1/sample_weight.shape[0])

  sample_weight = sample_weight/np.sum(sample_weight)

  factor_pos_p = sample_weight[np.logical_and(y==1,X[:,sa_index]==sa_value)][0] / (1/sample_weight.shape[0])
  factor_pos_np = sample_weight[np.logical_and(y==1,X[:,sa_index]!=sa_value)][0] / (1/sample_weight.shape[0])
  factor_neg_p =  sample_weight[np.logical_and(y!=1,X[:,sa_index]==sa_value)][0] / (1/sample_weight.shape[0])
  factor_neg_np =  sample_weight[np.logical_and(y!=1,X[:,sa_index]!=sa_value)][0] / (1/sample_weight.shape[0])

  factor_pos_p = 2*factor_pos_p*len(y_G_1[y_G_1==1]) / (2*factor_pos_p*len(y_G_1[y_G_1==1])+len(y))
  factor_pos_np = 2*factor_pos_np*len(y_G_2[y_G_2==1]) / (2*factor_pos_np*len(y_G_2[y_G_2==1])+len(y))
  factor_neg_p = 2*factor_neg_p*len(y_G_1[y_G_1==-1]) / (2*factor_neg_p*len(y_G_1[y_G_1==-1])+len(y))
  factor_neg_np = 2*factor_neg_np*len(y_G_2[y_G_2==-1]) / (2*factor_neg_np*len(y_G_2[y_G_2==-1])+len(y))
  
  print([factor_pos_p,factor_pos_np,factor_neg_np,factor_neg_p])

  return min([factor_pos_p,factor_pos_np,factor_neg_np,factor_neg_p])

def get_max_fairness_factor(X,y,saIndex,saValue):
  y_G_1 = y[X[:,saIndex] == saValue]
  y_G_2 = y[X[:,saIndex] != saValue]
  num_group = min([len(y_G_1[y_G_1==1]),len(y_G_1[y_G_1==-1]),len(y_G_2[y_G_2==1]),len(y_G_2[y_G_2==-1])])
  total_num = len(y)
  return 2*num_group/(2*num_group+total_num)


def val_baseline(X, y, sa_index, p_Group,method="",folders=10,test_size=0.5,random_state=20,n_estimators=40,depth=1):
  data_store = pd.DataFrame(columns=('Fairness_factor', 'Acc', 'Acc_G1', 'Acc_G2', 'Largest_Eq_Od', 'Ratio_Eq_Od', 'Eq_Od_Sum', 'FNR', 'FPR','DP_Dif','DP_Ratio','FNR_G1', 'FPR_G1','FNR_G2', 'FPR_G2'))
  from sklearn.ensemble import AdaBoostClassifier

  sss = StratifiedShuffleSplit(n_splits=folders, test_size=test_size, random_state=random_state)
  ave_acc,ave_acc_g1,ave_acc_g2, ave_eq_od, ave_eq_od_ratio,ave_eq_od_sum,ave_dp_dif,ave_dp_ratio= 0,0,0,0,0,0,0,0
  ave_fpr_g1,ave_fpr_g2,ave_fnr_g1,ave_fnr_g2 = 0,0,0,0
  ave_fp, ave_fn = 0,0
  for train_index, test_index in sss.split(X, y):
    from sklearn.tree import DecisionTreeClassifier
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),n_estimators=n_estimators, algorithm="SAMME", random_state=random_state)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    if method == 'EXG':
      from fairlearn.reductions import ExponentiatedGradient, DemographicParity
      np.random.seed(random_state)
      constraint = DemographicParity()
      mitigator = ExponentiatedGradient(classifier, constraint)
      sensitive_features = X_train[:,sa_index]
      print('EXG')
      mitigator.fit(X_train,y_train,sensitive_features=sensitive_features)
      y_pred_labels = mitigator.predict(X_test)
    elif method == "LFR":
      lfr = LFR(saIndex=sa_index,saValue=p_Group)
      lfr.fit(X_train,y_train)
      X_pred, y_pred_labels = lfr.transform(X_test,y_test)
    elif method == 'DRO':
      from DRO import train
      model = train(X_train,y_train, sa_index, p_Group)
      X_test_tensor = torch.Tensor(X_test).cuda()
      y_pred_labels = model(X_test_tensor)
      y_pred_labels =  y_pred_labels.argmax(dim=1).cpu().numpy()
    elif method == 'FTL':
      clf = FairRandomForestClassifier(
         orthogonality=0.5
      )
      s_train = X_train[:,sa_index]
      clf.fit(X_train,y_train,s_train)
      y_pred_labels = clf.predict(X_test)
    elif method == 'AdaFair':
      y_train[y_train == 0] = -1
      y_test[y_test == 0] = -1
      from AdaFair import AdaFair as AdaFair_origin
      clf = AdaFair_origin(n_estimators=n_estimators, saIndex=sa_index, saValue=p_Group, CSB="CSB2",cumul=True,use_validation=False)
      clf.fit(X_train,y_train)
      y_pred_labels = clf.predict(X_test)
    elif method == 'Adaboost':
      from sklearn.ensemble import AdaBoostClassifier
      clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth),n_estimators=n_estimators, algorithm="SAMME")
      clf.fit(X_train,y_train)
      y_pred_labels = clf.predict(X_test)
    elif method == "SMOTEBoost":
      from Competitors.SMOTEBoost import SMOTEBoost
      samples=100
      clf = SMOTEBoost(n_estimators=n_estimators,base_estimator=DecisionTreeClassifier(max_depth=depth),saIndex=sa_index,n_samples=samples,saValue=p_Group,CSB="CSB1")
      clf.fit(X_train,y_train)
      y_pred_labels = clf.predict(X_test)
    elif method == "Margin":
      from Competitors.margin import marginAnalyzer,boostingMarginAnalyzer
      clf = boostingMarginAnalyzer(X_train,y_train,protectedValue=p_Group,protectedIndex=sa_index,numRounds=n_estimators)
      clf.fit(X_train,y_train)
      y_pred_labels = clf.predict(X_test)
    elif method == 'GBDT':
      from sklearn.ensemble import GradientBoostingClassifier
      clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1, max_depth=depth, random_state=20)
      clf.fit(X_train, y_train)
      y_pred_labels = clf.predict(X_test)
    elif method == 'FairBatch':
      from FairBatch import train
      model = train(X_train,y_train, sa_index, p_Group)
      X_test_tensor = torch.Tensor(X_test).cuda()
      y_pred_labels = model(X_test_tensor)
      y_pred_labels = (F.tanh(y_pred_labels)+1)/2
      y_pred_labels[y_pred_labels>0.5] = 1
      y_pred_labels[y_pred_labels<=0.5] = 0
      y_pred_labels = y_pred_labels.cpu().detach().numpy()
    else:
      raise ValueError('No Baseline Type')

    acc = accuracy_score(y_pred_labels,y_test)
    gm = MetricFrame(metrics=accuracy_score, y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]))
    eq_od= equalized_odds_difference(y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]), method='between_groups', sample_weight=None)
    eq_od_ratio= equalized_odds_ratio(y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]), method='between_groups', sample_weight=None)
    FNR = false_negative_rate(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test)
    FPR = false_positive_rate(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test)
    dp_dif = demographic_parity_difference(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test,sensitive_features=X_test[:,sa_index])
    dp_ratio = demographic_parity_ratio(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test,sensitive_features=X_test[:,sa_index])
    gm_fpr = MetricFrame(metrics=false_positive_rate, y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]))
    gm_fnr = MetricFrame(metrics=false_negative_rate, y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]))

    # print(type(gm_fpr.by_group))       
    fpr_g1,fpr_g2 = gm_fpr.by_group.values[0],gm_fpr.by_group.values[1]
    fnr_g1,fnr_g2 = gm_fnr.by_group.values[0],gm_fnr.by_group.values[1]
    acc_g1,acc_g2 = gm.by_group.values[0],gm.by_group.values[1]
    ave_acc_g1 += acc_g1
    ave_acc_g2 += acc_g2
    ave_acc += acc
    ave_eq_od += eq_od
    ave_eq_od_ratio += eq_od_ratio
    ave_eq_od_sum += np.abs(fpr_g1-fpr_g2)+np.abs(fnr_g1-fnr_g2)
    ave_fp += FPR
    ave_fn += FNR
    ave_dp_dif += dp_dif
    ave_dp_ratio += dp_ratio
    ave_fpr_g1 += fpr_g1
    ave_fpr_g2 += fpr_g2
    ave_fnr_g1 += fnr_g1
    ave_fnr_g2 += fnr_g2


  ave_acc_g1 = ave_acc_g1/folders
  ave_acc_g2 = ave_acc_g2/folders
  ave_acc = ave_acc/folders
  ave_eq_od = ave_eq_od/folders
  ave_eq_od_ratio = ave_eq_od_ratio/folders
  ave_eq_od_sum = ave_eq_od_sum/folders
  ave_fp = ave_fp/folders
  ave_fn = ave_fn/folders
  ave_dp_dif = ave_dp_dif/folders
  ave_dp_ratio = ave_dp_ratio/folders
  ave_fpr_g1,ave_fpr_g2 = ave_fpr_g1/folders,ave_fpr_g2/folders
  ave_fnr_g1,ave_fnr_g2 = ave_fnr_g1/folders,ave_fnr_g2/folders

  group_acc = (ave_acc_g1+ave_acc_g1)/2
  balanced_acc = (1-ave_fp+1-ave_fn)/2
  group_balanced_acc = (1-ave_fpr_g1+1-ave_fpr_g2+1-ave_fnr_g1+1-ave_fnr_g2)/4

  data_store = data_store.append([{'Acc':ave_acc, 'Acc_G1':ave_acc_g1, 'Acc_G2':ave_acc_g2, 'Largest_Eq_Od':ave_eq_od, 'Ratio_Eq_Od':ave_eq_od_ratio, 'Eq_Od_Sum':ave_eq_od_sum, 'FNR':ave_fn, 'FPR':ave_fp,'DP_Dif':ave_dp_dif,'DP_Ratio':ave_dp_ratio,'FNR_G1':ave_fnr_g1, 'FPR_G1':ave_fpr_g1,'FNR_G2':ave_fnr_g2, 'FPR_G2':ave_fpr_g2, 'G_ACC':group_acc, 'B_ACC':balanced_acc, 'G_B_ACC':group_balanced_acc}],ignore_index=True)

  print("----------------")
  print(method)
  print("Acc",ave_acc)
  print("Acc_G1",ave_acc_g1)
  print("Acc_G2",ave_acc_g2)
  print("Lagest Equal Odd",ave_eq_od)
  print("Equal Odd Ratio",ave_eq_od_ratio)
  print("Equal Odd Sum",ave_eq_od_sum)
  print("Ave FN",ave_fn)
  print("Ave FP",ave_fp)
  print("Ave DP Dif",ave_dp_dif)
  print("Ace DP Ratio",ave_dp_ratio)
  print('group acc',group_acc)
  print('balanced acc',balanced_acc)
  print('group balanced acc',group_balanced_acc)
  return data_store



def val_algorithm_group_adaboost(X, y, sa_index, p_Group, x_control,Al="AdaBoost",folders=10,test_size=0.5,random_state=20,n_estimators=40,fairness_factors=[0.2,0.5,0.8,1.0,1.2],imbalance=True,acc_iter_ratio=0,use_group_acc=False,modify_d=False,acc_stop=False,check_point=False,use_alpha_acc=False,balance_monitor=False,depth=1,boost_metric='Acc',adaptive_weight=False,tau_weight=[0.25,0.25,0.25,0.25],learning_rate=1,norm_each_group=True,boosting_type='post_hoc'):
  data_store = pd.DataFrame(columns=('Fairness_factor', 'Acc', 'Acc_G1', 'Acc_G2', 'Largest_Eq_Od', 'Ratio_Eq_Od', 'Eq_Od_Sum', 'FNR', 'FPR','DP_Dif','DP_Ratio','FNR_G1', 'FPR_G1','FNR_G2', 'FPR_G2'))
  print(Al)

  # Only Try Different Fairness Factors for our methods
  if Al == "AdaFair" or Al == "AdaBoost":
    fairness_factors = [0]
  for fairness_factor in fairness_factors:
    print('Fairness Factor: '+str(fairness_factor))
    sss = StratifiedShuffleSplit(n_splits=folders, test_size=test_size, random_state=random_state)
    ave_acc,ave_acc_g1,ave_acc_g2, ave_eq_od, ave_eq_od_ratio,ave_eq_od_sum,ave_dp_dif,ave_dp_ratio= 0,0,0,0,0,0,0,0
    ave_fpr_g1,ave_fpr_g2,ave_fnr_g1,ave_fnr_g2 = 0,0,0,0
    ave_fp, ave_fn = 0,0
    for train_index, test_index in sss.split(X, y):
      if Al == "Ours":
        classifier = Group_AdaBoost(n_estimators=n_estimators, saIndex=sa_index, saValue=p_Group, imbalance=imbalance, alpha_restri=True, modify_d=modify_d, fairness_factor=fairness_factor, acc_iter_ratio=acc_iter_ratio, group_acc=use_group_acc,acc_stop=acc_stop,check_point=check_point,use_alpha_acc=use_alpha_acc,balance_monitor=balance_monitor,depth=depth,boost_metric=boost_metric,adaptive_weight=adaptive_weight,tau_weight=tau_weight,learning_rate=learning_rate,norm_each_group=norm_each_group,boosting_type=boosting_type)
      elif Al == "AdaFair":
        classifier = AdaFair_origin(n_estimators=n_estimators, saIndex=sa_index, saValue=p_Group, cumul=True , CSB="CSB1")
      elif Al == "AdaBoost":
        classifier = AdaBoostClassifier(n_estimators=n_estimators, algorithm="SAMME")
      # print('round')
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      classifier.fit(X_train, y_train)
      y_pred_probs = classifier.predict_proba(X_test)
      y_pred_labels = classifier.predict(X_test)
      acc = accuracy_score(y_pred_labels,y_test)
      gm = MetricFrame(metrics=accuracy_score, y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]))
      eq_od= equalized_odds_difference(y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]), method='between_groups', sample_weight=None)
      eq_od_ratio= equalized_odds_ratio(y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]), method='between_groups', sample_weight=None)
      FNR = false_negative_rate(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test)
      FPR = false_positive_rate(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test)
      dp_dif = demographic_parity_difference(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test,sensitive_features=X_test[:,sa_index])
      dp_ratio = demographic_parity_ratio(y_pred=y_pred_labels.reshape(-1,1),y_true=y_test,sensitive_features=X_test[:,sa_index])
      gm_fpr = MetricFrame(metrics=false_positive_rate, y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]))
      gm_fnr = MetricFrame(metrics=false_negative_rate, y_true=pd.Series(y_test[:,0]), y_pred=y_pred_labels, sensitive_features=pd.Series(X_test[:,sa_index]))

      # print(type(gm_fpr.by_group))       
      fpr_g1,fpr_g2 = gm_fpr.by_group.values[0],gm_fpr.by_group.values[1]
      fnr_g1,fnr_g2 = gm_fnr.by_group.values[0],gm_fnr.by_group.values[1]
      acc_g1,acc_g2 = gm.by_group.values[0],gm.by_group.values[1]
      ave_acc_g1 += acc_g1
      ave_acc_g2 += acc_g2
      ave_acc += acc
      ave_eq_od += eq_od
      ave_eq_od_ratio += eq_od_ratio
      ave_eq_od_sum += (1+eq_od_ratio)*eq_od
      ave_fp += FPR
      ave_fn += FNR
      ave_dp_dif += dp_dif
      ave_dp_ratio += dp_ratio
      ave_fpr_g1 += fpr_g1
      ave_fpr_g2 += fpr_g2
      ave_fnr_g1 += fnr_g1
      ave_fnr_g2 += fnr_g2


    ave_acc_g1 = ave_acc_g1/folders
    ave_acc_g2 = ave_acc_g2/folders
    ave_acc = ave_acc/folders
    ave_eq_od = ave_eq_od/folders
    ave_eq_od_ratio = ave_eq_od_ratio/folders
    ave_eq_od_sum = ave_eq_od_sum/folders
    ave_fp = ave_fp/folders
    ave_fn = ave_fn/folders
    ave_dp_dif = ave_dp_dif/folders
    ave_dp_ratio = ave_dp_ratio/folders
    ave_fpr_g1,ave_fpr_g2 = ave_fpr_g1/folders,ave_fpr_g2/folders
    ave_fnr_g1,ave_fnr_g2 = ave_fnr_g1/folders,ave_fnr_g2/folders

    group_acc = (ave_acc_g1+ave_acc_g1)/2
    balanced_acc = (1-ave_fp+1-ave_fn)/2
    group_balanced_acc = (1-ave_fpr_g1+1-ave_fpr_g2+1-ave_fnr_g1+1-ave_fnr_g2)/4

    data_store = data_store.append([{'Fairness_factor':fairness_factor, 'Acc':ave_acc, 'Acc_G1':ave_acc_g1, 'Acc_G2':ave_acc_g2, 'Largest_Eq_Od':ave_eq_od, 'Ratio_Eq_Od':ave_eq_od_ratio, 'Eq_Od_Sum':ave_eq_od_sum, 'FNR':ave_fn, 'FPR':ave_fp,'DP_Dif':ave_dp_dif,'DP_Ratio':ave_dp_ratio,'FNR_G1':ave_fnr_g1, 'FPR_G1':ave_fpr_g1,'FNR_G2':ave_fnr_g2, 'FPR_G2':ave_fpr_g2, 'G_ACC':group_acc, 'B_ACC':balanced_acc, 'G_B_ACC':group_balanced_acc}],ignore_index=True)

    print("----------------")
    print(Al)
    print("Fairness_factor",fairness_factor)
    print("Acc",ave_acc)
    print("Acc_G1",ave_acc_g1)
    print("Acc_G2",ave_acc_g2)
    print("Lagest Equal Odd",ave_eq_od)
    print("Equal Odd Ratio",ave_eq_od_ratio)
    print("Equal Odd Sum",ave_eq_od_sum)
    print("Ave FN",ave_fn)
    print("Ave FP",ave_fp)
    print("Ave DP Dif",ave_dp_dif)
    print("Ace DP Ratio",ave_dp_ratio)
    print('group acc',group_acc)
    print('balanced acc',balanced_acc)
    print('group balanced acc',group_balanced_acc)
  return data_store


def convert_data(X, y, sa_index, p_Group,data_num_list=[2500,1000,500,200]):
  saIndex,saValue = sa_index,p_Group
  X_G_p = X[X[:,saIndex] == saValue,:]
  y_G_p = y[X[:,saIndex] == saValue]
  X_G_np = X[X[:,saIndex] != saValue,:]
  y_G_np = y[X[:,saIndex] != saValue]

  X_G_p_pos = X_G_p[y_G_p == 1][:data_num_list[0]]
  X_G_p_neg = X_G_p[y_G_p != 1][:data_num_list[1]]
  X_G_np_pos = X_G_np[y_G_np == 1][:data_num_list[2]]
  X_G_np_neg = X_G_np[y_G_np != 1][:data_num_list[3]]

  y_G_p_pos = y_G_p[y_G_p == 1][:data_num_list[0]]
  y_G_p_neg = y_G_p[y_G_p != 1][:data_num_list[1]]
  y_G_np_pos = y_G_np[y_G_np == 1][:data_num_list[2]]
  y_G_np_neg = y_G_np[y_G_np != 1][:data_num_list[3]]


  X = np.concatenate([X_G_p_pos,X_G_p_neg,X_G_np_pos,X_G_np_neg],axis=0)
  y = np.concatenate([y_G_p_pos,y_G_p_neg,y_G_np_pos,y_G_np_neg])

  import random
  random.seed(20)

  idx = list(range(X.shape[0]))
  random.shuffle(idx)
  # print(idx)

  X = X[idx]
  y = y[idx]


  print(X.shape)
  print(y.shape)
  
  return X,y

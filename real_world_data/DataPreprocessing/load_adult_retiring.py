import folktables
from folktables import adult_filter
import numpy as np
from folktables import ACSDataSource, ACSEmployment

def load_retiring_adult(survey_year='2018',horizon='1-Year',survey='person',loc='CA'):
  ACSIncomeNew = folktables.BasicProblem(
    features=[
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 25000,    
    group='SEX',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
    )
  
  data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)
  acs_data = data_source.get_data(states=[loc], download=True)
  features, labels, groups = ACSIncomeNew.df_to_numpy(acs_data)
  x_control = {'SEX':groups}

  y = np.random.rand(labels.shape[0])
  y[labels] = 1
  y[labels==False] = -1
  p_Group = groups[0]


  for i in range(features.shape[-1]):
    if (features[:,i] == groups).all():
      sa_index = i

  print("----Class-----")
  print(len(labels))
  print("Pos ",len(y[y==1]))
  print("Neg ",len(y[y==-1]))
  print("Ratio ",len(y[y==-1])/len(y[y==1]))

  print("----Group-----")
  print("P ",len(groups[groups==p_Group]))
  print("NP ",len(groups[groups!=p_Group]))

  
  return features, y, sa_index, p_Group, x_control


def analysis_adult(survey_year='2018',horizon='1-Year',survey='person',loc='CA'):
  ACSIncomeNew = folktables.BasicProblem(
    features=[
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 25000,    
    group='SEX',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
    )
  
  data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)
  acs_data = data_source.get_data(states=[loc], download=True)
  features, labels, groups = ACSIncomeNew.df_to_numpy(acs_data)
  x_control = {'SEX':groups}

  y = np.random.rand(labels.shape[0])
  y[labels] = 1
  y[labels==False] = -1
  p_Group = groups[0]


  for i in range(features.shape[-1]):
    if (features[:,i] == groups).all():
      sa_index = i

  print("----Class-----")
  print(len(labels))
  print("Pos ",len(y[y==1]))
  print("Neg ",len(y[y==-1]))
  print("Ratio ",len(y[y==-1])/len(y[y==1]))

  Pos = len(y[y==1])
  Neg = len(y[y==-1])
  Ratio = Pos/Neg

  print("----Group-----")
  print("P ",len(groups[groups==p_Group]))
  print("NP ",len(groups[groups!=p_Group]))
  
  P_group = len(groups[groups==p_Group])
  NP_group = len(groups[groups!=p_Group])
  P_NP_ratio = P_group/NP_group

  dic = {'Loc':loc,'Pos':Pos,'Neg':Neg,'Ratio':Ratio,'P_group':P_group,'NP_group':NP_group,'P_NP_ratio':P_NP_ratio}
  

  return dic

from load_adult_retiring import analysis_adult

import pandas as pd
"""
loc_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
data_list = []
for loc in loc_list:
    data_list.append(analysis_adult(loc=loc,survey_year='2018'))

import pandas as pd

data = pd.DataFrame(data_list)
data.to_csv('/localscratch/xuezhiyu/fairness/AdaFair/2018_adult.csv')
"""

data = pd.read_csv('/localscratch/xuezhiyu/fairness/AdaFair/2018_adult.csv')

data = data[data['Ratio']-1<0.5]
print(data)
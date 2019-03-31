# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:31:19 2019

@author: bharat
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


import os




os.chdir('G:\\DataScience\\School of AI\\pslovedata')


Period=pd.read_csv('Period.csv')


Symptom=pd.read_csv('Symptom.csv')

User=pd.read_csv('User.csv')


data_1_1.merge(data_1_2_gropby,left_on='Hourly_Sports', right_on='Hourly_Sports')


data_1=User.merge(Period,left_on='id',right_on='User_id')


data_1.columns



data_final=data_1.merge(Symptom,left_on='User_id',right_on='user_id')


Symptom.columns





data_final.columns

data_final.pop('user_id')



data_final.User_id==data_final.user_id


np.unique(data_final.User_id==data_final.user_id)



data_final.isna().sum()



a=data_final[(data_final.end_date.isna())&(data_final.start_date.isna())]





data_final.start_date==np.nan

np.unique(data_final.end_date==np.nan)



col=data_final.columns.tolist()
columns = [c for c in col if c not in ["dob","start_date", "end_date","User_id","date"]]








for col in columns:
    sns.distplot(data_final[col])
    plt.show()



%matplotlib inline




for col in columns:
    data_final[col].plot.hist()
    plt.title(col)
    plt.show()


col=Symptom.columns.tolist()
columns = [c for c in col if c not in ["dob","start_date", "end_date","User_id","date"]]





for col in columns:
    sns.distplot(Symptom[col])
    plt.show()




for col in columns:
    Symptom[col].plot.hist()
    plt.title(col)
    plt.show()





data_sym=Symptom.copy()



data_sym.columns


data_sym.pop('date')




records = []  
for i in range(0, 13512):  
    records.append([str(data_sym.values[i,j]) for j in range(0, 10)])





l_new =[]            
for lis_outter in records:
    l_temp=[]
    for val in lis_outter :
        if val != 'nan':
            l_temp.append(val)
    l_new.append(l_temp) 
l=l_new


records=l_new


records_new=[]

for l in records:
    final_list = []
    for num in l: 
        if num not in final_list: 
            final_list.append(num) 
    
    records_new.append(final_list)
    




from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


  
te=TransactionEncoder()
te_data=te.fit(records_new).transform(records_new)
data_x=pd.DataFrame(te_data,columns=te.columns_)
print(data_x.head())

frequent_items= apriori(data_x, use_colnames=True, min_support=0.0045)

rules = association_rules(frequent_items, metric="lift", min_threshold=1)



rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

rules.columns



x =rules.support
y =rules.confidence
z =rules.lift






fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label support')
ax.set_ylabel('Y Labelconfidence')
ax.set_zlabel('z Label lift')

plt.show()


rules.plot.scatter(x='lift', y='confidence')





















# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 00:42:25 2019

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


final_data=Symptom.merge(User,left_on='user_id',right_on='id')
final_data.columns

final_data.columns

final_data.pop('date')

final_data.replace(0,np.nan,inplace=True)

labels = ["{0} - {1}".format(i, i + 4) for i in range(1, 101, 5)]



col=final_data.columns.tolist()

for cl in col:
    final_data[cl] = pd.cut(final_data[cl], range(1, 106, 5), right=False, labels=labels)

final_data=final_data.astype(str)

final_data.replace('nan','no_issue',inplace=True)



for cl in col:
    final_data[cl]=final_data[cl]+'_'+str(cl)



records = []  
for i in range(0, 13512):  
    records.append([str(final_data.values[i,j]) for j in range(0, 12)])




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

rules = association_rules(frequent_items, metric="lift", min_threshold=)



rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))















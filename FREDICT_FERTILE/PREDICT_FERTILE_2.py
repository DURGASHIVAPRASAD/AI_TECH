# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:13:26 2019

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


data_sym=Symptom.copy()



data_sym.columns


data_sym.pop('date')



data_sym.replace(0,np.nan,inplace=True)



labels = ["{0} - {1}".format(i, i + 4) for i in range(1, 101, 5)]

data_sym.columns


data_sym['group'] = pd.cut(data_sym.acne, range(1, 106, 5), right=False, labels=labels)


data_sym.pop('group')

data_sym.columns



data_sym['group'] = pd.cut(data_sym.acne, range(1, 106, 5), right=False, labels=labels)



col=data_sym.columns.tolist()

for cl in col:
    data_sym[cl] = pd.cut(data_sym[cl], range(1, 106, 5), right=False, labels=labels)
    

data_sym=data_sym.astype(str)


data_sym.replace('nan','no_issue',inplace=True)

data_sym.info()


data_sym['dd']=data_sym.acne+"acne"

data_sym.pop('dd')

for cl in col:
    data_sym[cl]=data_sym[cl]+'_'+str(cl)


%matplotlib inline


for cl in data_sym:
    data_sym[cl].value_counts().sort_values(ascending=True).plot.barh()
    plt.show()



for cl in data_sym:
    data_sym[data_sym[cl]!='no_issue_'+str(cl)][cl].value_counts().sort_values(ascending=True).plot.barh()
    plt.show()
    data_sym[data_sym[cl]=='no_issue_'+str(cl)][cl].value_counts().sort_values(ascending=True).plot.barh()
    plt.show()














data_sym.acne.value_counts().sort_values(ascending=True).plot.barh()


pd.Series.value_counts





data_User=User.copy()


data_User.columns


data_User.pop('dob')













for cl in data_User:
    data_User[cl] = pd.cut(data_User[cl], range(1, 106, 5), right=False, labels=labels)


data_User=data_User.astype(str)
    

for col in data_User:
    data_User[col].value_counts().sort_values(ascending=True).plot.barh()
    plt.title(str(col))
    plt.show()



data_User.cycle_length_initial.unique()






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



data_User['id']=User.id

data_sym['User_id']=Symptom.user_id



data_final=data_sym.merge(data_User,left_on='User_id',right_on='id')

data_final.columns

data_final.cycle_length_initial=data_final.cycle_length_initial+'_'+'cycle_length'

data_final.period_length_initial=data_final.period_length_initial+'_'+'period_length'



data_final.columns


data_final_1=data_final.copy()



data_final_1.columns


data_final_1.pop('id')





records = []  
for i in range(0, 13512):  
    records.append([str(data_final_1.values[i,j]) for j in range(0, 12)])






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



























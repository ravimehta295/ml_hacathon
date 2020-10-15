#!/usr/bin/env python
# coding: utf-8

# #Data cleaning

# In[1]:


import pandas as pd
import math
import numpy as np
file_name = "train.xlsx"
df = pd.read_excel(io=file_name)


# In[2]:


df.head()


# In[3]:


#df.drop(columns='ID', inplace = True)


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


for item in df.columns:
    print(item, df[item].count())


# In[7]:


#df.sort_values(by='Number_of_Weeks_Used', inplace=True, ascending=False, na_position='last')


#  #replacing clay as 1 and slit as 0
#  #replacing food crop as 1 and dfeed crop as 0

# In[8]:


#df.tail() 


# In[9]:



train_data =[]
null_data =[]
count = 0

for index in range(0,len(df['Crop'])):
    temp_soil = 0
    if df['Soil'][index] == 'clay':
        temp_soil= 1
    else:
        temp_soil= 0
    temp_crop =0
    if df['Crop'][index] == 'Food':
        temp_crop = 1
    else:
        temp_crop = 0
    if pd.isna(df['Number_of_Weeks_Used'][index] ):
        null_data.append([df['Insects'][index],temp_soil,
                          temp_crop,df['Category_of_Toxicant'][index] ,df['Does_count'][index] ,df['Number_of_Weeks_Used'][index] 
                        , df['Number_Weeks_does_not used'][index] , df['Season'][index] , df['Crop_status'][index] ])
    else:
        train_data.append([df['Insects'][index],temp_soil,
                          temp_crop,df['Category_of_Toxicant'][index] ,df['Does_count'][index] ,df['Number_of_Weeks_Used'][index] 
                        , df['Number_Weeks_does_not used'][index] , df['Season'][index] , df['Crop_status'][index] ])


# In[11]:


train_not_null = pd.DataFrame(train_data,columns =df.columns[1:])
train_null = pd.DataFrame(null_data,columns =df.columns[1:])


# In[12]:


train_not_null.head()


# In[13]:


file_name = "test.xlsx"
test = pd.read_excel(io=file_name)


# In[14]:


test.head()


# In[15]:


for item in test.columns:
    print(item, test[item].count())


# In[16]:



test_train_data =[]
test_null_data =[]
count = 0

for index in range(0,len(df['Crop'])):
    temp_soil = 0
    if df['Soil'][index] == 'clay':
        temp_soil= 1
    else:
        temp_soil= 0
    temp_crop =0
    if df['Crop'][index] == 'Food':
        temp_crop = 1
    else:
        temp_crop = 0
    if pd.isna(df['Number_of_Weeks_Used'][index] ):
        test_null_data.append([df['Insects'][index],temp_soil,
                          temp_crop,df['Category_of_Toxicant'][index] ,df['Does_count'][index] ,df['Number_of_Weeks_Used'][index] 
                        , df['Number_Weeks_does_not used'][index] , df['Season'][index] , df['Crop_status'][index] ])
    else:
        test_train_data.append([df['Insects'][index],temp_soil,
                          temp_crop,df['Category_of_Toxicant'][index] ,df['Does_count'][index] ,df['Number_of_Weeks_Used'][index] 
                        , df['Number_Weeks_does_not used'][index] , df['Season'][index] , df['Crop_status'][index] ])


# In[17]:


import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# In[18]:


#y_tr = np.array(train_not_null['Crop_status'])
#X_tr = np.array(train_not_null.drop(['Crop_status'], axis=1))


#y_test = np.array(train_not_null['Crop_status'])
#X_test = np.array(train_not_null.drop(['Crop_status'], axis=1))
X_tr, X_test, y_tr, y_test = sk.model_selection.train_test_split(train_not_null.drop(['Crop_status'], axis=1), train_not_null['Crop_status'], test_size=0.2, shuffle=True, random_state=36)


# In[19]:


LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, y_tr)
LR.predict(X_test)
round(LR.score(X_test,y_test), 4)


# In[20]:


SVM = svm.SVC(decision_function_shape="ovo").fit(X_tr, y_tr)
SVM.predict(X_test)
round(SVM.score(X_test, y_test), 4)


# In[21]:


RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_tr, y_tr)
RF.predict(X_test)
round(RF.score(X_test, y_test), 4)


# In[22]:


NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_tr, y_tr)
NN.predict(X_test)
round(NN.score(X_test, y_test), 4)


# In[ ]:





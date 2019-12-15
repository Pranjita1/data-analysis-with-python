#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd


# In[50]:


df = pd.read_csv('bank.csv')


# In[51]:


df.head(10)


# In[52]:


df.dropna()


# In[53]:


df.tail(20)


# In[54]:


df.info()
df.replace(to_replace = 'yes', value = 1, inplace = True)
df.replace(to_replace = 'no', value = 0, inplace = True)
df['marital'].replace(to_replace = 'married',value = 1, inplace = True)
df['marital'].replace(to_replace = 'divorced',value = 2, inplace = True)
df['education'].replace(to_replace = 'secondary',value = 1, inplace = True)
df['education'].replace(to_replace = 'tertiary',value = 2, inplace = True)
df['marital'].replace(to_replace = 'single',value = 0, inplace = True)
df['education'].replace(to_replace = 'primary',value = 0, inplace = True)
df.replace(to_replace = 'unknown', value = 3, inplace = True)
df.replace(to_replace = 'failure', value = 4, inplace = True)


# In[55]:


df.corr()


# In[56]:


z = df[['age', 'marital', 'education', 'default','duration', 'balance', 'housing', 'loan']]
y = df['deposit']


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn


# In[58]:


z_train,z_test,y_train,y_test = train_test_split(z,y,test_size=0.25,random_state=0)


# In[59]:


lr = LogisticRegression()


# In[60]:


lr.fit(z_train, y_train)


# In[61]:


lr.score(z_test, y_test)


# In[62]:


lr.predict(z_test)


# In[63]:


y_pred = lr.predict(z_test)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)


# In[64]:


print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))


# In[65]:


from sklearn.neighbors import KNeighborsClassifier


# In[66]:


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(z_train, y_train)


# In[67]:


knn.predict(z_test)


# In[68]:


knn.score(z_test, y_test)


# In[69]:


ypred = knn.predict(z_test)
confusion_matrix = pd.crosstab(y_test, ypred)


# In[70]:


sn.heatmap(confusion_matrix, annot = True)


# In[71]:


cust_new = df.drop(['job','day','month','duration','campaign','pdays','previous','poutcome'], axis = 1)
cust_new[0:10]


# In[72]:


cust_new.describe()


# In[73]:


cust_new['age'] = (cust_new['age'] > 40.5).astype(float)
cust_new['balance'] = (cust_new['balance'] > 1000).astype(float)
cust_new.head(20)


# In[75]:


cust_new.to_csv(r'C:\Users\HitBuy Atrium\Desktop\cust.csv')


# In[76]:


def parse_file(name):
    df = pd.read_csv(name, sep=',')
    df = df.replace(to_replace='positive', value=1)
    df = df.replace(to_replace='negative', value=0)
    
    x = np.array(df['deposit'])
    del df['deposit']
    bin_df = dummy_encode_categorical_columns(df)
    return np.array(bin_df).astype(int), x


# In[77]:


import copy
import numpy as np
def dummy_encode_categorical_columns(data):
    result_data = copy.deepcopy(data)
    for column in data.columns.values:
        result_data = pd.concat([result_data, pd.get_dummies(result_data[column], prefix = column, prefix_sep = ': ')], axis = 1)
        del result_data[column]
    return result_data
X_train, y_train = parse_file('traincust.csv')
X_test, y_test = parse_file('testcust.csv')

X_train_pos = X_train[y_train == 1]
X_train_neg = X_train[y_train == 0]


# In[78]:



y_pred = []
for test_obj in X_test:
    pos = np.sum(test_obj == X_train_pos) / float(len(X_train_pos))
    neg = np.sum(test_obj == X_train_neg) / float(len(X_train_neg))
    if (pos > neg):
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[79]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


# In[80]:


print("Accuracy: {}\nPrecision: {}\nRecall: {}".format(acc, prec, rec))


# In[ ]:





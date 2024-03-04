#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bank_data = pd.read_csv("C:/Users/91992/Downloads/bank+marketing/bank-additional/bank-additional/bank-additional.csv", delimiter=';')
bank_data.rename(columns={'y':"deposit"}, inplace=True)


# In[3]:


bank_data.head()


# In[4]:


bank_data.tail()


# In[5]:


bank_data.shape
bank_data.columns
bank_data.dtypes


# In[7]:


bank_data.dtypes.value_counts()
bank_data.info()


# In[8]:


bank_data.duplicated().sum()


# In[9]:


bank_data.describe() 


# In[10]:


bank_data.describe(include="object")


# In[12]:


bank_data.hist(figsize=(10,10),color="blue")
plt.show()


# In[13]:


categorical = bank_data.select_dtypes(include='object').columns
print(categorical)

numerical = bank_data.select_dtypes(exclude='object').columns
print(numerical)


# In[15]:


for  attribute in categorical:
    plt.figure(figsize=(10,10))  
    sns.countplot(x=attribute, data=bank_data, color="green")
    plt.title(f'Bar Plot of {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()


# In[16]:


bank_data.plot(kind="box",subplots=True,figsize=(30,15),color="red",layout=(2,5))
plt.show()


# In[17]:


Col = bank_data[['age','campaign','duration']]
Q1 = np.percentile(Col, 25)
Q3 = np.percentile(Col, 75)
IQR = Q3 - Q1
Lower_bound = Q1 - 1.5 * IQR
Upper_bound = Q3 + 1.5 * IQR
bank_data[['age','campaign','duration']] = Col[(Col > Lower_bound) & (Col < Upper_bound)]


# In[18]:


bank_data.plot(kind="box",subplots=True,figsize=(30,15),color="purple",layout=(2,5))
plt.show()


# In[19]:


numeric_columns = bank_data.select_dtypes(include=['number'])
corr = numeric_columns.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.2)
plt.title("Correlation Heatmap")
plt.show()


# In[20]:


High_Correlation = ["emp.var.rate","euribor3m","nr.employed"]


# In[21]:


bank_data1 = bank_data.copy()
bank_data1.columns


# In[22]:


bank_data1.drop(High_Correlation,inplace=True,axis=1)  # axis=1 indicates columns
bank_data1.columns


# In[23]:


bank_data1.shape


# In[24]:


from sklearn.preprocessing import LabelEncoder
Label_encoding = LabelEncoder()
bank_data_encoded = bank_data1.apply(Label_encoding.fit_transform)
bank_data_encoded


# In[25]:


bank_data_encoded['deposit'].value_counts()


# In[26]:


x = bank_data_encoded.drop('deposit',axis=1) 
y = bank_data_encoded['deposit']
print(y.shape)
print(type(x))
print(type(y))


# In[27]:


from sklearn.model_selection import train_test_split
print(4119*0.25)


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[29]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def eval_model(y_test,y_pred):
    Accuracy = accuracy_score(y_test,y_pred)
    print('Accuracy_Score',Accuracy)
    Confusion_mat = confusion_matrix(y_test,y_pred)
    print('Confusion Matrix\n',Confusion_mat)
    print('Classification Report\n',classification_report(y_test,y_pred))

def mscore(model):
    train_score = model.score(x_train,y_train)
    test_score = model.score(x_test,y_test)
    print('Training Score',train_score)  
    print('Testing Score',test_score)    


# In[30]:


from sklearn.tree import DecisionTreeClassifier
Decision_Tree = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=10)
Decision_Tree.fit(x_train,y_train)


# In[31]:


mscore(Decision_Tree)


# In[32]:


ypred_Decision_Tree = Decision_Tree.predict(x_test)
print(ypred_Decision_Tree)


# In[33]:


eval_model(y_test,ypred_Decision_Tree)


# In[34]:


from sklearn.tree import plot_tree

class_name = ['no','yes']
feature_name = x_train.columns
print(class_name)
print(feature_name)


# In[38]:


feature_name = bank_data.columns.tolist()
plt.figure(figsize=(15, 12))
plot_tree(Decision_Tree, feature_names=feature_name, filled=True)
plt.show()


# In[39]:


Decision_Tree1 = DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_split=15)
Decision_Tree1.fit(x_train,y_train)


# In[40]:


mscore(Decision_Tree1)


# In[41]:


ypred_Decision_Tree1 = Decision_Tree1.predict(x_test)


# In[42]:


eval_model(y_test,ypred_Decision_Tree1)


# In[43]:


plt.figure(figsize=(15,15))
plot_tree(Decision_Tree1,feature_names=feature_name,class_names=class_name,filled=True)
plt.show()


# In[ ]:





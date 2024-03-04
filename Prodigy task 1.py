#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


df = pd.read_csv("C:/Users/91992/Downloads/API_SP.POP.TOTL_DS2_en_csv_v2_85/API_SP.POP.TOTL_DS2_en_csv_v2_85.csv")
df


# In[31]:


df.head()


# In[32]:


df.tail()


# In[33]:


df.shape


# In[34]:


df.columns


# In[35]:


df.dtypes


# In[36]:


df.info()


# In[37]:


df.describe()


# In[38]:


df.duplicated().sum()


# In[39]:


df.isna().sum().any()
df = df.fillna(method="ffill")
df.head()


# In[40]:


df.isna().sum().any()


# In[41]:


df["Country Name"].unique()
df["Country Code"].unique()
df["Indicator Name"].unique()
df["Indicator Code"].unique()


# In[42]:


df.drop(["Indicator Name","Indicator Code","Country Code"], axis = 1, inplace = True)

df.columns


# In[43]:


Cols = ['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']


# In[44]:


for i in Cols:
        fig = plt.figure(figsize=(5,5))
        plt.hist(df[i],color='blue',bins=10)
        plt.xlabel(i)
        plt.show()


# In[45]:


years = df.columns[1:]


# In[46]:


Total = df[years].sum()


# In[48]:


plt.figure(figsize=(30,30))
plt.barh(years,Total,color='green')
plt.xlabel("Total of the country")
plt.ylabel("Year",size=20)
plt.title("Total count per year",size=20)
plt.show()


# In[49]:


country_by_1960 = df.sort_values(by='1960').head(10)
country_by_1960


# In[50]:


country_by_1960_t = country_by_1960.set_index("Country Name").T
for country_name,data_values in country_by_1960_t.iterrows():
    fig = plt.figure(figsize=(10,5))
    sns.barplot(x=data_values.index,y=data_values.values)
    plt.xlabel("Countries")
    plt.ylabel("Data Values")
    plt.title(f"{country_name} - Data Values from 1960 to 2022")
    plt.xticks(rotation=90)
    plt.show()


# In[51]:


country_by_2022 = df.sort_values(by='2022').head(10)
country_by_2022


# In[52]:


country_by_2022_t = country_by_2022.set_index("Country Name").T
for country_name,data_values in country_by_2022_t.iterrows():
    fig = plt.figure(figsize=(10,5))
    sns.barplot(x=data_values.index,y=data_values.values)
    plt.xlabel("Year")
    plt.ylabel("Data Value")
    plt.title(f"{country_name} - Data Values from 1960 to 2022")
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:





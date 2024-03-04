#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


Accidents = pd.read_csv("C:/Users/91992/Downloads/archive (1)/US_Accidents_March23.csv")


# In[3]:


Accidents.head()


# In[4]:


Accidents.tail()


# In[5]:


null_values = Accidents.isnull().sum()
print(null_values)


# In[6]:


Accidents.info()


# In[7]:


Accidents['Start_Time'] = pd.to_datetime(Accidents['Start_Time'], format='ISO8601')
Accidents['Hour'] = Accidents['Start_Time'].dt.hour
time_of_day_counts = Accidents['Hour'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.bar(time_of_day_counts.index, time_of_day_counts.values)
plt.xlabel('Hour of the Day')
plt.ylabel('Accident Count')
plt.title('Accidents by Time of Day')
plt.show()


# In[8]:


weather_counts = Accidents['Weather_Condition'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=weather_counts.index, y=weather_counts.values)
plt.xticks(rotation=90)
plt.xlabel('Weather Conditions')
plt.ylabel('Accident Count')
plt.title('Accidents by Weather Conditions')
plt.show()


# In[9]:


import random

def random_palette():
    """
    Create a random palette each every time
    """
    # Creating a mix of multiple palettes
    base_palette = sns.color_palette("pastel", 5) + sns.color_palette("tab20c", 5) + sns.color_palette("tab20b", 5)
    # Sample specific colors from the base palette
    colour_list = random.sample(base_palette, 5)
    # Blend the palettes to create a new palette
    palette = sns.blend_palette([colour for colour in colour_list], 10)
    return palette


# In[10]:


def subplot(df, i, column, order=None, pallete=random_palette()):
    plt.subplot(2, 3 ,i)
    sns.barplot(data=Accidents[column].value_counts().reset_index(), x = column, y = 'count', palette=random_palette(), order=order)
    plt.xticks(rotation=45)


# In[20]:


sns.set_style('whitegrid')
plt.figure(figsize=(15, 8))
plt.suptitle("Accident Frequency by Category", fontsize=17)

subplot(Accidents, i=1, column = 'Astronomical_Twilight')

age_order = ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown']
subplot(Accidents, i=2, column = 'Amenity', order= age_order)

day_order = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
subplot(Accidents, i=3, column = 'Bump', order=day_order)

education_order= ["Illiterate", "Writing & reading", "Elementary school", "Junior high school","High school","Above high school","Unknown"]

subplot(Accidents, i=4, column = 'Give_Way', order=education_order)

subplot(Accidents, i=5, column = 'Traffic_Signal')
subplot(Accidents, i=6, column = 'Turning_Loop')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
plt.show()


# In[ ]:


contributing_factors_counts = Accidents['Description'].value_counts()
plt.bar(contributing_factors_counts.index, contributing_factors_counts.values)
plt.xticks(rotation=90)
plt.xlabel('Contributing Factors')
plt.ylabel('Accident Count')
plt.title('Contributing Factors to Accidents')
plt.show()


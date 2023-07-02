# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:16:27 2023

@author: Franz
"""

# --------------------------------------------------------------

### Getting started

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # -> plt.show to plot the respective graph
import seaborn as sns
sns.set() # to change the theme

# show all columns in output
pd.options.display.max_columns=None

# read data from downloaded file
filename=r"C:\Users\Franz.000\Documents\GitHub\MAY23_BDA_INT_Crowdfunding\data\kaggle\Kaggle_Dataset.csv"
raw=pd.read_csv(filename)
raw.head(3)
df=raw

# raw["launched_year"]=pd.to_datetime(raw["launched_at"])
# raw["year"]=raw["launched_year"].dt.year
# raw["year"].value_counts()

# ------------------------------------------------------------------------------


### First inspection

## general overview
# df information
df.info() # 378661 projects

# detect (and delete duplicates)
df.duplicated().sum() # 0
df.id.duplicated().sum() # 22,170
df.drop_duplicates(keep='first',inplace=True,subset='id')
df.id.duplicated().sum() #
len(df) # 192,888 remain

# What categories does the success indicator "status" have?
df.status.value_counts(normalize=True)
sns.countplot(data=df,x="status")
plt.title("Frequencies of status categories");

# reduce to successful and failed projects, (finished projects) only
df=df[(df.status=="successful")|(df.status=="failed")]
len(df) # 180,675 are relevant and retained (after dropping duplicates)
df.status.value_counts(normalize=True) 


## date/time related information
# convert date variables into datetime format, and
# derive year, month, week, and day (of the week) 
from datetime import timedelta
for p in ["year","month","week","wday"]:
    for v,var in zip(["launched","dead"],["launched_at","deadline"]):
        name=v+"_"+p
        # convert into datetime
        df[v]=pd.to_datetime(df[var])
        # derive each of the periods 
        if p=="year":   df[name]=df[v].dt.year
        if p=="month":  df[name]=df[v].dt.month
        if p=="week":   df[name]=df[v].dt.week
        if p=="wday":    df[name]=(df[v].dt.dayofweek)+1 # +1 to make the week start (more readable) with day 1 (instead of 0)
        # print(df[name].value_counts(normalize=True)) # retain day of week at launch

## date-related distributions   
# year        
# original webrobots data frame contained outliers before 2009 (launch of kickstarter platform)
df.launched_year.value_counts() # but: outliers before 2009 already dropped with reduction to finished projects
print(pd.crosstab(df.launched_year,df.status,normalize="index"))
sns.displot(x=df.launched_year,hue=df.status)
plt.xlabel("year a campaign was launched")
plt.title("Distribution of campaign launches over years by status");

# month of launch     
sns.displot(x=df.launched_month,hue=df.status)
plt.xlabel("month a campaign was launched")
plt.title("Distribution of campaign launches over months by status");

# weekday
# first assign string instead of numeric values 
days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
df["launched_day"]=df["launched_wday"].replace(range(1,8),days)
# order according to order of a week
df["launched_day"]=pd.Categorical(df["launched_day"],categories=days,ordered=True)            
sns.countplot(x=df.launched_day,hue=df.status)
plt.xlabel("weekday a campaign was launched")
plt.title("Distribution of campaign launches over weekdays by status");

# duration 
df.duration.value_counts()
sns.displot(df.duration,kde=True)
plt.ylim(0,10000);
plt.xlabel("duration in days")
plt.title("Distribution of campaign durations");


## Main and sub categories
df.describe(include="object")

# reverse column names of category variables
new={"main_category":"sub_category",
"sub_category":"main_category"}
df.rename(columns=new,inplace=True)

# main category
pd.DataFrame(df.main_category.value_counts(normalize=True)[:10])
df.main_category.value_counts()

# sub category
pd.DataFrame(df.sub_category.value_counts(normalize=True))
df.sub_category.value_counts()

# What are the contents (sub categories) of each main category?
for cat in df.main_category.value_counts().index.tolist():
    print("\n",cat,"has following sub categories:\n",df.loc[df["main_category"]==cat].sub_category.value_counts(normalize=True).to_frame())
    
# (sub) category
df.sub_category.value_counts()[:10]
df.sub_category.nunique() # 159 modalities are way too many


## backers - plot calculation takes way too long
# sns.displot(x=df.backers_count,hue=df.status)
# plt.xlabel("number of backers a campaign attracted")
# plt.title("Distribution of backers by status");


## goal - plot calculation takes way too long
# sns.displot(x=df.goal_usd,hue=df.status)
# sns.displot(x=df.goal_usd)
# plt.xlabel("goal height of a campaign")
# plt.title("Distribution of goal height by status");
# plt.title("Distribution of goal height");

df[["backers_count","goal_usd"]].describe()
df.backers_count.value_counts().sort_index(ascending=False)[:20]


## creator
# count projects per creator 
df_creator=pd.DataFrame(df['creator_id'].value_counts())
df_creator['id']=df_creator.index
df_creator['creator_projects']=df_creator['creator_id']
df_creator.drop("creator_id",axis=1,inplace=True)
df_creator.rename(columns={"id":"creator_id"},inplace=True)

# Building groups
df_creator['creator_projects'].replace(1,'1',inplace=True)
df_creator['creator_projects'].replace(list(range(2,6)),'2-5',inplace=True)
df_creator['creator_projects'].replace(list(range(6,15)),'6-14',inplace=True)
df_creator['creator_projects'].replace(list(range(15,30)),'15-29',inplace=True)
df_creator['creator_projects'].replace(list(range(30,100)),'30+',inplace=True)
df_creator['creator_projects'].value_counts()

# merge to main dataframe
df=df.merge(right=df_creator,on="creator_id",how="inner")


## (pre)select features (columns) to retain for further processing
# define selection
list_final=['id','currency','backers_count','country','status', 'usd_pledged','main_category','creator_projects','goal_usd', 'city','launched_year','launched_day','duration']


# reduce dataset to selection
# df_final=df.id
# df_final=pd.concat([df_final,df[list_final]],axis=1)
# df_final=df_final.set_index("id")
# df_final.info()

df_final=df[list_final]
df_final.info()


#---------------------------------------------------------------

### SAVE FILE (directly in GitHub project)
filename=r'data\kaggle\Kaggle_deduplicated.csv'
f=open(filename,'w',encoding='utf-8')
f.write(df_final.to_csv())
f.close()

#---------------------------------------------------------------

from pathlib import Path # check file existence
path=Path(filename)
if path.is_file(): print("finished explore")
else: print("ERROR: file save failed")
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:16:27 2023

@author: Franz
"""

# ------------------------------------------------------------------------------

### Getting started

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # -> plt.show to plot the respective graph
import seaborn as sns
sns.set() # to change the theme

# read data from downloaded file
file=r"C:\Users\Franz.000\Documents\Berufliches\Weiterbildung\data analyst\data\kickstarter\kaggle\ks-projects-201801.csv"
raw=pd.read_csv(file)
raw.head(3)
df=raw

raw["launched_year"]=pd.to_datetime(raw["launched"])
raw["year"]=raw["launched_year"].dt.year
raw["year"].value_counts()

# ------------------------------------------------------------------------------


### First inspection

## general overview
# df information
df.info() # 378661 projects

# What categories does the success indicator "state" have?
df.state.value_counts(normalize=True)
sns.countplot(data=df,x="state")
plt.title("Frequencies of state categories")

# reduce to successful and failed projects, (finished projects) only
df=df[(df.state=="successful")|(df.state=="failed")]
len(df) # 331675 are relevant and retained
df.state.value_counts(normalize=True) 


# check for duplicates
df.duplicated().sum() # 0
df.ID.duplicated().sum() # 0

# check for missing values
df.isna().sum().sum() # 213
df.isna().sum() # only on variables (probably) not relevant



## date/time related information
# convert date variables into datetime format, and
# derive year, month, week, and day (of the week) 
from datetime import timedelta
for p in ["year","month","week","day"]:
    for v,var in zip(["launch","end"],["launched","deadline"]):
        name=v+"_"+p
        # convert into datetime
        df[v]=pd.to_datetime(df[var])
        # derive each of the periods 
        if p=="year":   df[name]=df[v].dt.year
        if p=="month":  df[name]=df[v].dt.month
        if p=="week":   df[name]=df[v].dt.week
        if p=="day":    df[name]=df[v].dt.dayofweek
        print(df[name].value_counts(normalize=True)) # retain day of week

        
# webrobots data frame contained outliers before 2009 (launch of kickstarter platform)
df.launch_year.value_counts() # but: outliers before 2009 already dropped with reduction to finished projects
print(pd.crosstab(df.launch_year,df.state,normalize="index"))

# derive duration of a campaign in different units
df['duration_s']=(df['end']-df['launch'])/timedelta(seconds=1)
df['duration_min']=(df['end']-df['launch'])/timedelta(minutes=1)
df['duration_h']=(df['end']-df['launch'])/timedelta(hours=1)
df['duration_d']=(df['end']-df['launch'])/timedelta(days=1)
df['duration_mon']=df['duration_s']/86400/30 # derive duration in months
df['duration_y']=df['duration_s']/86400/30/12 # derive duration in years
df[['duration_h','duration_d','duration_mon','duration_y']].describe() # duration_h and duration_d should be sufficient -> drop else
df.info()


## Main and sub categories
# main category
pd.DataFrame(df.main_category.value_counts(normalize=True))
df.main_category.value_counts()

# What are the contents (sub categories) of each main category?
for cat in df.main_category.value_counts().index.tolist():
    print("\n",cat,"has following sub categories:\n",df.loc[df["main_category"]==cat].category.value_counts(normalize=True).to_frame())
    
# (sub) category
df.category.value_counts()[:10]
df.category.nunique() # 159 modalities are way too many



## (pre)select features (columns) to retain for further processing
# define selection
list_active=["backers","main_category",
             "country","usd_goal_real","usd_pledged_real",
             "launch_year","end_year","launch_day","end_day",
             "duration_h","duration_d"]
# reduce dataset to selection
df_active=df[list_active]
df_active.info()


# ------------------------------------------------------------------------------

### Visualization

# (no?) correlations betwwen explanatory variables
sns.heatmap(df.corr(),cmap='winter')
plt.title('Correlation of all variables');
cols=['backers','usd_pledged_real','usd_goal_real','duration_h','duration_d',]
sns.pairplot(df[cols],diag_kind='kde')
plt.title('Correlation of all variables');




# Development of funding goals by project (top 3 of main) category and state over time
select=["Film & Video","Music","Publishing"]
g=sns.catplot(x="launched_year",y="goal",hue="state",row="main_category",\
            row_order=select,col="state",kind="bar",errorbar=('ci',False),height=4,data=df);
g.set_axis_labels("Year of project launch", "Funding goal")
g.set_xticklabels([],rotation=30,ha="right")
# add margin to top of plot grid
g.fig.subplots_adjust(top=0.93)
# modify titles of subplots
g.set_titles("{row_name} ({col_name})")
# add general grid title
g.fig.suptitle("Development of funding goals by project (top 3 of main) category and state over time");
plt.show()


# All the currency values are converted into appropriate amounts in USD as per the fx rate provided in the base dataset.
# --> we should use the _real columns for pledged and goal to be able to compare amounts across countries
df.columns
cols=['backers','usd_pledged_real','usd_goal_real','duration_h','duration_d',]
df[cols].head(10)
df[cols].describe()
for col in cols:
    sns.displot(df[col],kde=True);
plt.show()
    
df[["usd_goal_real","usd_pledged_real"]].head(10)
df[["usd_goal_real","usd_pledged_real"]].describe()
df["usd_goal_real"].value_counts()[:10]
df["usd_pledged_real"].value_counts()[:10]

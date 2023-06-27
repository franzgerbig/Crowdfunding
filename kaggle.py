# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:16:27 2023

@author: Franz
"""

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # -> plt.show to plot the respective graph
import seaborn as sns
sns.set() # to change the theme

# read data from downloaded file
file=r"C:\Users\Franz.000\Documents\Berufliches\Weiterbildung\data analyst\data\kickstarter\kaggle\ks-projects-201801.csv"
df=pd.read_csv(file)
df.head(3)

# general overview
df.info() # 378661 projects

# What categories does the success indicator "state" have?
df.state.value_counts(normalize=True)

# reduce to successful and failed projects, (finished projects) only
df=df[(df.state=="successful")|(df.state=="failed")]
len(df) # 331675 are relevant and retained
df.state.value_counts(normalize=True) 

# check for duplicates
df.duplicated().sum() # 0
df.ID.duplicated().sum() # 0


# convert date variables
import datetime
df.launched.value_counts()[:5] # type object
df['dead']=pd.to_datetime(df['deadline'])
df['launch']=pd.to_datetime(df['launched'])
def get_year(x):
    return x.year
df['launched_year']=df['launch'].apply(get_year)

# webrobots data frame contained outliers before 2009 (launch of kickstarter platform)
df.launched_year.value_counts() # outliers before 2009 already dropped with reduction to finished projects


# derive duration of a campaign in different units
from datetime import timedelta
df['duration_s']=(df['dead']-df['launch'])/timedelta(seconds=1)
df['duration_min']=(df['dead']-df['launch'])/timedelta(hours=1)
df['duration_d']=(df['dead']-df['launch'])/timedelta(days=1)
df['duration_mon']=df['duration_s']/86400/30 # derive duration in months
df['duration_y']=df['duration_s']/86400/30/12 # derive duration in years
df[['duration_d','duration_mon','duration_y']].describe() # duration_d is sufficient
df.info()
df.launched_year.value_counts() # outliers for 1970 must be dropped
print(pd.crosstab(df.launched_year,df.state,normalize="index"))

# category
df.category.value_counts()[:10]
df.category.nunique() # 159 modalities
# df[["goal","pledged","usd pledged","usd_pledged_real","usd_goal_real"]].describe()
# plt.boxplot(df["goal"])
# sns.relplot(x="goal",y="backers",data=df,kind="line")
# sns.displot(df.goal,kde=True,bins=15)
# # df["usd pledged"].value_counts()

# main category
pd.DataFrame(df.main_category.value_counts(normalize=True))


### Visualization

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
# --> we should use the real columns for pledged and goal to be able to compare amounts across countries
df[["usd pledged","usd_pledged_real"]].head(3)
df[["usd pledged","usd_pledged_real"]].describe()

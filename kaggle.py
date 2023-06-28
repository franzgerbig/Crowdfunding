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

# show all columns in output
pd.options.display.max_columns=None

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
list_active=["backers",
             "main_category",
             "country","usd_goal_real","usd_pledged_real",
             "launch_year","launch_day",
             "duration_h","duration_d",
             "state"]
# list of maybe-variables
list_unclear=["end_year","end_day"]

# reduce dataset to selection
df_active=df.ID
df_active=pd.concat([df_active,df[list_active]],axis=1)
df_active=df_active.set_index("ID")
df_active.info()


# ------------------------------------------------------------------------------

### Preprocessing

## 
# check for duplicates
df_active.duplicated().sum() # 0
df_active.index.duplicated().sum() # 0

# check for missing values
df_active.isna().sum().sum() # 0

# check for outliers
df_active.describe().T
for col in df_active.select_dtypes("number"):
    print("\n",col,":\n",df_active[col].value_counts())
# no outliers found
df_active.country.value_counts() # 210 campaigns of country 'N,0"'
df_active.country.value_counts(normalize=True) # <0.1 % -> drop
df_active=df_active.loc[df_active["country"]!='N,0"']

# however, duration_d may be recoded (starting with 1 instead of 0)
df_active["launch_day"]=df_active["launch_day"]+1
df_active["launch_day"].value_counts()




## standardization (since variables don't share same scale)
# first of all: split into train and test set
from sklearn.model_selection import train_test_split
X=df_active.drop("state",axis=1)
y=df_active.state
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

# standardization
df_active.info()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# select variables to scale - date variables should not be scaled
scale_cols=['backers','usd_pledged_real','usd_goal_real']
X_train[scale_cols]=scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols]=scaler.transform(X_test[scale_cols])
X_train.head()
X_train.isna().sum()


## encodings
# encode explanatory variables

# main category
df_active.main_category.value_counts() # simply OHE

# country
df_active.country.value_counts() # divide into US, GB/CA, rest

# encode in a loop over train and test set
for s in [X_train,X_test]:
    # main category
    s=pd.get_dummies(s,columns=["main_category"],prefix="",prefix_sep="")
    
    # country
    s.loc[s["country"]=="US","country_US"]=1
    s.loc[(s["country"]=="GB")|(s["country"]=="CA"),"country_GB_CA"]=1
    s.loc[~s["country"].isin(["US","GB","CA"]),"country_rest"]=1
    # fill NAs with 0s
    s[["country_US","country_GB_CA","country_rest"]]=s[["country_US","country_GB_CA","country_rest"]].fillna(0)
    s[["country_US","country_GB_CA","country_rest"]].value_counts()
    for c in ["country_US","country_GB_CA","country_rest"]:
        print("\n",c,":\n",pd.crosstab(s.country,s[c]))
        print(s[c].value_counts())
    print(s.info())
    print(s.isna().sum().sum())


# encode target variable
for s in [y_train,y_test]:
    # main category
    s=pd.get_dummies(s,columns=["state"],prefix="",prefix_sep="")
    print(s.value_counts())
    print(s.isna().sum().sum())



# ------------------------------------------------------------------------------

### Visualization

## BEFORE preprocessing
# (no?) correlations betwwen explanatory variables
sns.heatmap(df_active.corr(),cmap='winter',annot=True)
plt.title('Correlation of all variables');
cols=['backers','usd_pledged_real','usd_goal_real','duration_h','duration_d']
sns.pairplot(df_active[cols],diag_kind='kde')
# sns.pairplot(df_active,diag_kind='kde')
plt.title('Relation of subset of variables');

# relation between backers and converted pledged amount
plt.scatter(df_active['backers'],df_active['usd_pledged_real'])
plt.plot((df_active["backers"].min(),df_active["backers"].max()),(df_active["usd_pledged_real"].min(),                                                             df_active["usd_pledged_real"].max()),"red")
plt.xlabel('backers count')
plt.ylabel('(converted) pledged amount')
plt.title('backers and pledged amount');


# Development of funding goals by project (top 3 of main) category and state over time
select=["Comics","Technology","Film & Video"]
g=sns.catplot(x="launched_year",y="backers_count",hue="state",row="cat_parent_name",\
            row_order=select,col="state",kind="bar",errorbar=('ci',False),height=4,data=df_active);
g.set_axis_labels("Year of project launch","Number of backers")
plt.tick_params(bottom='off',labelbottom='on')
# plt.xticks(rotation=30) # does it only for the very last subplot
g.set_xticklabels(rotation=30,ha="right")
# add margin to top of plot grid (to havc enough space for grid title)
g.fig.subplots_adjust(top=.93)
# modify titles of subplots
g.set_titles("{row_name} ({col_name})")
# add general grid title
g.fig.suptitle("Development of backers' quantity by project (top 3 of main) category and state over time");


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

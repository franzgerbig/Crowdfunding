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
file=r"C:\Users\Franz.000\Documents\GitHub\MAY23_BDA_INT_Crowdfunding\data\kaggle\Kaggle_Dataset.csv"
raw=pd.read_csv(file)
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
len(df) # 180675 are relevant and retained (after dropping duplicates)
df.status.value_counts(normalize=True) 


## date/time related information
# convert date variables into datetime format, and
# derive year, month, week, and day (of the week) 
from datetime import timedelta
for p in ["year","month","week","day"]:
    for v,var in zip(["launched","dead"],["launched_at","deadline"]):
        name=v+"_"+p
        # convert into datetime
        df[v]=pd.to_datetime(df[var])
        # derive each of the periods 
        if p=="year":   df[name]=df[v].dt.year
        if p=="month":  df[name]=df[v].dt.month
        if p=="week":   df[name]=df[v].dt.week
        if p=="day":    df[name]=(df[v].dt.dayofweek)+1 # +1 to make the week start (more readable) with day 1 (instead of 0)
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
sns.displot(x=df.launched_day,hue=df.status)
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
    print("\n",cat,"has following sub categories:\n",df.loc[df["main_category"]==cat].category.value_counts(normalize=True).to_frame())
    
# (sub) category
df.category.value_counts()[:10]
df.category.nunique() # 159 modalities are way too many


## backers
# sns.displot(x=df.backers_count,hue=df.status)
# plt.xlabel("number of backers a campaign attracted")
# plt.title("Distribution of backers by status");


## goal
# sns.displot(x=df.goal_usd,hue=df.status)
sns.displot(x=df.goal_usd)
plt.xlabel("goal height of a campaign")
# plt.title("Distribution of goal height by status");
plt.title("Distribution of goal height");



## (pre)select features (columns) to retain for further processing
# define selection
list_final=['id', 'currency', 'backers_count', 'country', 'status', 'usd_pledged','main_category', 'creator_id', 'goal_usd', 'city', 'launched_year','duration']

# ['id', 'currency', 'backers_count', 'country', 'status', 'usd_pledged','sub_category', 'creator_id', 'goal_usd', 'city', 'launched_year','duration']


# reduce dataset to selection
# df_final=df.id
# df_final=pd.concat([df_final,df[list_final]],axis=1)
# df_final=df_final.set_index("id")
# df_final.info()

df_final=df[list_final]
df_final.info()

# ------------------------------------------------------------------------------

### Preprocessing

## 

# check for missing values
df_final.isna().sum().sum() # 0

# check for outliers
df_final.describe().T
# numeric variables
for col in df_final.select_dtypes("number"):
    print("\n",col,":\n",df_final[col].value_counts())
# no outliers found

# categorical variable country
df_final.country.value_counts() # no outliers
df_final.country.value_counts(normalize=True) #



## standardization (since variables don't share same scale)
# first of all: split into train and test set
from sklearn.model_selection import train_test_split
X=df_final.drop("status",axis=1)
y=df_final.status
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

# standardization
df_final.info()
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
df_final.main_category.value_counts() # simply OHE

# country
df_final.country.value_counts() # divide into US, GB/CA, rest

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
    s=pd.get_dummies(s,columns=["status"],prefix="",prefix_sep="")
    print(s.value_counts())
    print(s.isna().sum().sum())



# ------------------------------------------------------------------------------

### Visualization

## BEFORE preprocessing
cols=['backers_count','usd_pledged','goal_usd','launched_year','duration']
# (no?) correlations betwwen explanatory variables
sns.heatmap(df_final[cols].corr(),cmap='winter',annot=True)
plt.title('Correlation of variables');

sns.pairplot(df_final[cols],diag_kind='kde')
plt.title('Distribution of variables');

# relation between backers and converted pledged amount
plt.scatter(df_final['backers_count'],df_final['usd_pledged'])
plt.plot((df_final["backers_count"].min(),df_final["backers_count"].max()),(df_final["usd_pledged"].min(),                                                             df_final["usd_pledged"].max()),"red")
plt.xlabel('number of backers')
plt.ylabel('pledged amount')
plt.title('backers and pledged amount');


# Development of funding goals by project (top 3 of main) category and status over time
select=["Comics","Technology","Film & Video"]
g=sns.catplot(x="launched_year",y="backers_count",hue="status",row="cat_parent_name",\
            row_order=select,col="status",kind="bar",errorbar=('ci',False),height=4,data=df_final);
g.set_axis_labels("Year of project launch","Number of backers")
plt.tick_params(bottom='off',labelbottom='on')
# plt.xticks(rotation=30) # does it only for the very last subplot
g.set_xticklabels(rotation=30,ha="right")
# add margin to top of plot grid (to havc enough space for grid title)
g.fig.subplots_adjust(top=.93)
# modify titles of subplots
g.set_titles("{row_name} ({col_name})")
# add general grid title
g.fig.suptitle("Development of backers' quantity by project (top 3 of main) category and status over time");


# ANOVA
num_list=['backers_count',
       'usd_pledged','goal_usd',
       'launched_year','duration']
import statsmodels.api
for item in num_list:
    print('###',item)
    result=statsmodels.formula.api.ols(f'{item} ~ status',data=df_final).fit()
    display(statsmodels.api.stats.anova_lm(result))
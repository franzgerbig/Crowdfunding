# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 14:03:10 2023

@authors: Hendrik Bosse, Franz Gerbig
"""



# ---------------------------------------------------------------

### Preprocessing

## settings
import pandas as pd
# show all columns in output
pd.options.display.max_columns=None

# load data created in exploration
filename=r"data\kaggle\Kaggle_deduplicated.csv"
df_final=pd.read_csv(filename,index_col='id')
df_final.info()
df_final.drop("Unnamed: 0",axis=1,inplace=True)

## quick check
# check for missing values
df_final.isna().sum().sum() # 0?


# check for outliers
df_final.describe().T

# numeric variables
for col in df_final.select_dtypes("number"):
    print("\n",col,":\n",df_final[col].value_counts())
# no outliers found

# categorical variable country
df_final.country.value_counts() # only 2 SI
df_final=df_final.loc[df_final["country"]!="SI"]
df_final.country.value_counts() # no more

# currency
df_final.currency.value_counts() # only 6 PLN
df_final=df_final.loc[df_final["currency"]!="PLN"]
df_final.currency.value_counts() # no more


## first of all: split into train and test set
# split 
from sklearn.model_selection import train_test_split
# separate explanatory and target variable(s)
X=df_final.drop("status",axis=1)
y=df_final['status'].replace(['successful','failed'],[1,0]) # on the fly: assign numerical values to target categories
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)


## standardization (since variables don't share same scale)
# standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# select variables to scale - date variables should not be scaled
scale_list=['backers_count','usd_pledged','goal_usd','duration']
X_train[scale_list]=scaler.fit_transform(X_train[scale_list])
X_test[scale_list]=scaler.transform(X_test[scale_list])
X_train_scale=X_train[scale_list]
X_test_scale=X_test[scale_list]
X_train_scale.head()


## encodings
# encode explanatory variables
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
cat_list=["currency","country","main_category","creator_projects",
 "launched_year","launched_day"]

# encode train set
ohe_train=ohe.fit_transform(X_train[cat_list])
column_name=ohe.get_feature_names_out(cat_list)
X_train_ohe=pd.DataFrame(ohe_train,columns=column_name,index=X_train.index)
X_train_ohe.info()

# encode test set
ohe_test=ohe.transform(X_test[cat_list])
column_name=ohe.get_feature_names_out(cat_list)
X_test_ohe=pd.DataFrame(ohe_test,columns=column_name,index=X_test.index)
X_test_ohe.info()

# bring 'em together
X_train_merge=pd.concat([X_train_scale,X_train_ohe],axis=1)
X_test_merge=pd.concat([X_test_scale,X_test_ohe],axis=1)
X_train_merge.head()


#---------------------------------------------------------------

### SAVE FILES (directly in GitHub project)    
    
# X_train
filename=r'data\kaggle\Kaggle_X_train_merge.csv'
f=open(filename,'w',encoding='utf-8') # if only write ('w') specified, existing file will be replaced
f.write(X_train_merge.to_csv())
f.close()

# X_test
filename=r'data\kaggle\Kaggle_X_test_merge.csv'
f=open(filename,'w',encoding='utf-8')
f.write(X_test_merge.to_csv())
f.close()

# y_train
filename=r'data\kaggle\Kaggle_y_train.csv'
f=open(filename,'w',encoding='utf-8')
f.write(y_train.to_csv())
f.close()

# y_test
filename=r'data\kaggle\Kaggle_y_test.csv'
f=open(filename,'w',encoding='utf-8')
f.write(y_test.to_csv())
f.close()

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:15:07 2023

@author: Franz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# simplification
def simple_backers():
    df['backers_count'].replace(range(0,101),'0-100',inplace=True)
    df['backers_count'].replace(range(101,1001),'101-1000',inplace=True)
    df['backers_count'].replace(range(1001,10001),'1001-10000',inplace=True)
    df['backers_count'].replace(range(10001,200000),'10000+',inplace=True)

def simple_country():
    df['country'].replace(['GB','CA'],'GB,CA',inplace=True)
    df['country'].replace(['DE','FR','IT','ES','NL','SE','DK','CH','IE','BE','AT','NO','LU','PL','GR','SI'],'Europe',inplace=True)
    df['country'].replace(['HK','SG','JP'],'Asia',inplace=True)
    df['country'].replace(['MX'],'South America and Mexico',inplace=True)
    df['country'].replace(['AU','NZ'],'Australia and new Zealand',inplace=True)

def simple_category():
    df['main_category'].replace(['music','film & video','games','comics','dance'],'Entertainment',inplace=True)
    df['main_category'].replace(['art','fashion','design','photography','theater'],'Culture',inplace=True)
    df['main_category'].replace(['technology','publishing','food','crafts','journalism'],'Others',inplace=True)
    
def simple_currency():
    df['currency'].replace(['AUD','MXN','SEK','HKD','NZD','DKK','SGD','CHF','JPY','NOK','PLN'],'Others',inplace=True)
    
    
# proprocessing
def preprocess():
    # Only Successful and failed projects are important for us
    df=df.loc[(df['status']=='successful')|(df['status']=='failed')]
    # drop duplicates
    df.drop_duplicates(keep='first',inplace=True,subset='id')
    
    # reverse column names of category variables correctly
    new={"main_category":"sub_category2",
    "sub_category":"main_category"}
    df.rename(columns=new,inplace=True)
    df.drop("sub_category2",axis=1,inplace=True)
    
    # variable selection
    cat_drop.append('sub_category')
    cat_drop.append('name')
    
    # count projects per creator 
    df_creator=pd.DataFrame(df_final['creator_id'].value_counts())
    df_creator['id']=df_creator.index
    df_creator['creator_project_counts']=df_creator['creator_id']
    df_creator.drop("creator_id",axis=1,inplace=True)
    df_creator.rename(columns={"id":"creator_id"},inplace=True)
    df_creator
    # Building groups
    df_creator.loc[df_creator['creator_project_counts']==1,'creator_projects']='1'
    df_creator.loc[(df_creator['creator_project_counts']>=2) & (df_creator['creator_project_counts']<=5),'creator_projects']='2-5'
    df_creator.loc[(df_creator['creator_project_counts']>=6) & (df_creator['creator_project_counts']<=15),'creator_projects']='6-15'
    df_creator.loc[(df_creator['creator_project_counts']>=16) & (df_creator['creator_project_counts']<=30),'creator_projects']='16-30'
    df_creator.loc[(df_creator['creator_project_counts']>=31),'creator_projects']='31+'
    # merge to main dataframe
    df_final=df_final.merge(right=df_creator,on="creator_id",how="inner")
    df_final.drop(['creator_id','creator_project_counts'],axis=1,inplace=True)
    
    # make status numerical
    df['status'].replace(['successful','failed'],[1,0],inplace=True)
    
    
    
    
    # prepare modeling
    
    # Splitting of the data
    # Target Variable
    df['status'].replace(['successful','failed'],[1,0],inplace=True)
    df_y=df['status']
        
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(df.drop(columns='status',axis=1),df_y,test_size=0.2,random_state=42)
        
    # standardizing of the numerical variables
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    X_train_num=X_train[num_list]
    X_test_num=X_test[num_list]
    X_train_num[num_list]=sc.fit_transform(X_train[num_list])
    X_test_num[num_list]=sc.fit_transform(X_test[num_list])
        
    # OneHotEncoding of the categorical variable of X_train
    from sklearn.preprocessing import OneHotEncoder
    ohe=OneHotEncoder(sparse=False)
    ohe_train=ohe.fit_transform(X_train[cat_list])
    column_name=ohe.get_feature_names_out(cat_list)
    X_train_ohe=pd.DataFrame(ohe_train,columns=column_name,index=X_train.index)
        
    # OHE X_test variable
    ohe2=OneHotEncoder(sparse=False)
    ohe_test=ohe2.fit_transform(X_test[cat_list])
    column_name2=ohe2.get_feature_names_out(cat_list)
    X_test_ohe=pd.DataFrame(ohe_test,columns=column_name2,index=X_test.index)
    X_train_merge_org=pd.concat([X_train_num,X_train_ohe],axis=1)
    X_test_merge=pd.concat([X_test_num,X_test_ohe],axis=1)
        
    # Resampling
    if over_estimation ==1:
        from imblearn.over_sampling import RandomOverSampler 
        rOs=RandomOverSampler()
        X_over,y_over=rOs.fit_resample(X_train_merge_org,y_train)
        X_train_merge=X_over
        y_train=y_over
    else:
        X_train_merge=X_train_merge_org
            
    if under_estimation==1:
        from imblearn.under_sampling import RandomUnderSampler
        rUs=RandomUnderSampler()
        X_under,y_under=rUs.fit_resample(X_train_merge_org,y_train)
        X_train_merge=X_under
        y_train=y_under
            

    # Potential one hot encoder problem
    # Some values are unique which cuases a mismatch in columns
    for item in X_train_merge.columns:
        if item not in X_test_merge.columns:
            print('Error',item)
            X_test_merge[item]=0
    for item in X_test_merge.columns:
        if item not in X_train_merge.columns:
            print('Error',item)
            X_train_merge[item]=0



# do modeling
# def modeling():
    
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:23:09 2023

@authors: Hendrik Bosse, Franz Gerbig
"""

# -----------------------------------------------------------------

### Optimize modeling

## settings
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# load data
X_train_merge=pd.read_csv("data/kaggle/Kaggle_X_train_merge.csv",index_col='id')
X_test_merge=pd.read_csv("data/kaggle/Kaggle_X_test_merge.csv",index_col='id')
y_train=pd.read_csv("data/kaggle/Kaggle_y_train.csv")
y_test=pd.read_csv("data/kaggle/Kaggle_y_test.csv")

# target data need to be one-dimensional!!!   
y_train.drop("id",axis=1,inplace=True)
y_test.drop("id",axis=1,inplace=True)
y_train=y_train.squeeze()
y_test=y_test.squeeze()


## may be better to group countries, since US are so overrepresented
# encode in a loop over train and test set (and for memory use)
for s in [X_train_merge,X_test_merge]:  
    # encode to 1
    s.loc[(s["country_GB"]==1)|(s["country_CA"]==1),"country_GB_CA"]=1
    s.loc[(s["country_US"]!=1)&(s["country_GB"]!=1)&(s["country_CA"]!=1),"country_rest"]=1
    # fill NAs with 0s
    s[["country_US","country_GB_CA","country_rest"]]=s[["country_US","country_GB_CA","country_rest"]].fillna(0)
    s[["country_US","country_GB_CA","country_rest"]].value_counts()
    for c in ["country_US","country_GB_CA","country_rest"]:
        print(s[c].value_counts())
    print(s.info())
    print(s.isna().sum().sum())

# drop dummies of countries grouped in country_rest    
drop_dummies=["country_AT",
"country_AU",
"country_BE",
"country_CA",
"country_CH",
"country_DE",
"country_DK",
"country_ES",
"country_FR",
"country_GB",
"country_GR",
"country_HK",
"country_IE",
"country_IT",
"country_JP",
"country_LU",
"country_MX",
"country_NL",
"country_NO",
"country_NZ",
"country_PL",
"country_SE",
"country_SG"]
for d in drop_dummies:
    X_train_merge.drop(d,axis=1,inplace=True)
    X_test_merge.drop(d,axis=1,inplace=True)

# due to computational/RAM limits: reduce data volume
for r in range(2009,2021):
    name="launched_year_"+str(r)
    X_train_merge.drop(name,axis=1,inplace=True)
    X_test_merge.drop(name,axis=1,inplace=True)
# for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
#     name="launched_day_"+d
#     X_train_merge.drop(name,axis=1,inplace=True)
#     X_test_merge.drop(name,axis=1,inplace=True)

# Moreover, backers_count seems to be by far most important, it's more an outcome than input variable, though.
# (At least) the latter is true for usd_pledged and duration
for o in ["backers_count","usd_pledged","duration"]:
    X_train_merge.drop(o,axis=1,inplace=True)
    X_test_merge.drop(o,axis=1,inplace=True)


## start new modeling
# import required methods
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from datetime import datetime

# select and instantiate model types
tree=DecisionTreeClassifier(random_state=0)
log=LogisticRegression(random_state=0)
rfc=RandomForestClassifier(random_state=0)
# prepare loop
models =    [tree,rfc,log]
models_=    ['tree','rfc','log']
compare=pd.DataFrame({"id":models_,
    #                   "MSE_train":"",
    #                   "RMSE_train":"",
    #                   "MSE_test":"",
    #                   "RMSE_test":"",                     
    #                   "Train_score":"",
    #                   "Test_score":""
                    }) 

# start loop of Grid Search of best parameters
for m,m_ in zip(models,models_):
    # GridSearch
    print("\nStarting the grid search for best",m_,"model")
    start_time=datetime.now()
    print("at:",start_time.strftime("%H:%M:%S"))
    
    # define model hyperparameters grid to search through for best combination
    if m_=='log':
        grid={'max_iter':range(100,201),
              'solver':["lbfgs","liblinear"]}
              # 'solver':["lbfgs","liblinear","newton-cg"]}
    else:
        grid={'max_depth':range(3,16),
              'criterion':["entropy","gini"]}

    # set grid parameters
    gs_name_=m_+"_gs" # to use as variable name   
    gs_name =m_+"_gs" # to create/refer to instance
    gs_name=GridSearchCV(estimator=m,param_grid=grid,
                      cv=3,n_jobs=-1,verbose=3)
    
    # train model according to prior defined (hyper)parameters
    gs_name.fit(X_train_merge,y_train)
    # store predictions for later use in evaluation (eg, classification report)
    pred_train_name="pred_train"+gs_name_
    pred_test_name="pred_test"+gs_name_
    pred_train_name=gs_name.predict(X_train_merge)
    pred_test_name=gs_name.predict(X_test_merge)
    
    # show best model
    # compare errors (RMSE)
    # compare.loc[(compare['id']==m_),"MSE_train"]=mean_squared_error(y_train, pred_train_name)
    # compare.loc[(compare['id']==m_),"RMSE_train"]= np.sqrt(mean_squared_error(y_train,pred_train_name))
    compare.loc[(compare['id']==m_),"MSE test"]=mean_squared_error(y_test,pred_test_name)
    compare.loc[(compare['id']==m_),"RMSE test"]= np.sqrt(mean_squared_error(y_test,pred_test_name))
    compare.loc[(compare['id']==m_),"Train score"]=gs_name.score(X_train_merge,y_train)
    compare.loc[(compare['id']==m_),"Test score"]=gs_name.score(X_test_merge,y_test)
    compare.loc[(compare['id']==m_),"Best params"]=str(gs_name.best_params_)

    # print("best parameters:\n",gs_name.best_params_)
    # print("train score:",gs_name.score(X_train_merge,y_train))
    # print("test score:",gs_name.score(X_test_merge,y_test))
    
    # Classification report
    print(classification_report(y_test,pred_test_name))
        
    # How long did it take?
    from time import time
    from datetime import timedelta
    elapsed=(datetime.now()-start_time)/timedelta(seconds=1)
    print("Best model parameters found after:",elapsed,"seconds")

# show metrics comparison 
compare.set_index("id",inplace=True)
compare.T # Transpose, since (now) more columns than rows
    

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
import matplotlib.pyplot as plt
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


# retained columns are
print("retained columns are:\n",X_train_merge.columns)
# 

## start new modeling
# import required methods
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from datetime import datetime
from datetime import timedelta
from time import time

# select and instantiate model types
tree=DecisionTreeClassifier(random_state=0)
rfc=RandomForestClassifier(random_state=0)
# log=LogisticRegression(random_state=0)
# prepare loop
models =    [tree,rfc]
models_=    ['tree','rfc']
compare=pd.DataFrame({"model":models_,
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
    start_time=datetime.now()
    print("\nStarting the grid search for best",m_,"model","at:",start_time.strftime("%H:%M:%S"))
    
    # define model hyperparameters grid to search through for best combination
    if m_=='log':
        grid={'max_iter':range(100,150),
              'solver':["sag"]}
              # 'solver':["lbfgs","liblinear","newton-cg"]}
    else:
        grid={'max_depth':range(3,20),
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
    
    # compare errors
    # compare.loc[(compare['id']==m_),"MSE_train"]=mean_squared_error(y_train, pred_train_name)
    # compare.loc[(compare['id']==m_),"RMSE_train"]= np.sqrt(mean_squared_error(y_train,pred_train_name))
    compare.loc[(compare['model']==m_),"MSE test"]=mean_squared_error(y_test,pred_test_name)
    compare.loc[(compare['model']==m_),"RMSE test"]= np.sqrt(mean_squared_error(y_test,pred_test_name))
    compare.loc[(compare['model']==m_),"Train score"]=gs_name.score(X_train_merge,y_train)
    compare.loc[(compare['model']==m_),"Test score"]=gs_name.score(X_test_merge,y_test)
    compare.loc[(compare['model']==m_),"Best params"]=str(gs_name.best_params_)
    elapsed=(datetime.now()-start_time)/timedelta(seconds=1)
    compare.loc[(compare['model']==m_),"Calculation time"]=elapsed
    
    # Classification report
    print(classification_report(y_test,pred_test_name))
        
    # How long did it take?
    print("Best model parameters found after:",elapsed,"seconds")


# show metrics comparison 
compare.set_index("model",inplace=True)
compare
    
# show and compare feature importances
# tree
# bar plot
tree=DecisionTreeClassifier(criterion='gini',
                             max_depth=9,random_state=0)
tree.fit(X_train_merge,y_train)    
tree_importances=pd.DataFrame({
    "Variables":X_train_merge.columns,
    "Importance":tree.feature_importances_
}).sort_values(by='Importance',ascending=False)
tree_importances.nlargest(5, "Importance").plot.bar(x="Variables",y="Importance",
                   figsize=(18, 5),color="#4529de");


# tree coefs as line plot
tree_importance=pd.DataFrame({'Variables':X_train_merge.columns,'Coefficient':tree.feature_importances_})
display(tree_importance.sort_values(by='Coefficient',ascending=False))
tree_importance_negatives=tree_importance.sort_values('Coefficient',ascending=True)
tree_importance_positives=tree_importance.sort_values('Coefficient',ascending=False)

# Plotting the Coefficeint DataFrame
plt.figure(figsize=[10,10])
plt.plot(tree_importance_positives['Variables'][0:5],tree_importance_positives['Coefficient'][:5])
plt.ylabel('Coefficient')
plt.xlabel('Variables')
plt.title('Top 5 positive coefficients');

plt.figure(figsize=[10,10])
plt.plot(tree_importance_negatives['Variables'][0:5],log_importance_negatives['Coefficient'][:5])
plt.ylabel('Coefficient')
plt.xlabel('Variables')
plt.title('Top 5 negative coefficients');
plt.show()


# rfc
rfc=RandomForestClassifier(criterion='entropy',
                           max_depth=15,random_state=0)    
rfc.fit(X_train_merge,y_train)    
feat_importances=pd.DataFrame({
    "Variables":X_train_merge.columns,
    "Importance":rfc.feature_importances_
}).sort_values(by='Importance',ascending=False)

# tree (ascending importance -> most negative coefs first)
tree.fit(X_train_merge,y_train)    
tree=DecisionTreeClassifier(criterion='gini',
                            max_depth=9,random_state=0)
feat_importances=pd.DataFrame({
    "Variables":X_train_merge.columns,
    "Importance":tree.feature_importances_
}).sort_values(by='Importance',ascending=True)
feat_importances.nlargest(5, "Importance").plot.bar(x="Variables",y="Importance",
                   figsize=(18, 5),color="#4529de");



# 
selected=['goal_usd','creator_projects_1','main_category_food']
tree.fit(X_train_merge[selected],y_train)    
pred_test=tree.predict(X_test_merge[selected])

# Classification report
print(classification_report(y_test,pred_test))

feat_importances.nlargest(5, "Importance").plot.bar(x="Variables",y="Importance",
                   figsize=(18, 5),color="#4529de");
feat_importances=pd.DataFrame({
    "Variables":X_train_merge[selected].columns,
    "Importance":tree.feature_importances_
}).sort_values(by='Importance',ascending=False)
    

# ----------------------------------------------------------------------

# Stop execution here
quit()

"""
# ----------------------------------------------------------------

### next steps

# backers_count as target
# group backers?
# and do multinomial logistic or alike?

df_final.backers_count.value_counts()[:50] # a cut into 3 groups should be enough
df_final.backers_count.describe()



## Split
# create new data split and new target
from sklearn.model_selection import train_test_split
# separate explanatory and target variable(s)
X=df_final.drop("backers_count",axis=1)
y=pd.qcut(df_final["backers_count"],3,labels=False) # on the fly: cut numerical target values into categories
X_train,X_test,y_train_opt,y_test_opt=train_test_split(X,y,test_size=.3,random_state=0)


## standardization (since variables don't share same scale)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# select variables to scale - date variables should not be scaled
scale_list=['goal_usd']
X_train[scale_list]=scaler.fit_transform(X_train[scale_list])
X_test[scale_list]=scaler.transform(X_test[scale_list])
X_train_scale=X_train[scale_list]
X_test_scale=X_test[scale_list]


## encodings
# encode explanatory variables
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
cat_list=["currency","country","main_category","creator_projects","launched_day"]

# encode train set
ohe_train=ohe.fit_transform(X_train[cat_list])
column_name=ohe.get_feature_names_out(cat_list)
X_train_ohe=pd.DataFrame(ohe_train,columns=column_name,index=X_train.index)

# encode test set
ohe_test=ohe.transform(X_test[cat_list])
column_name=ohe.get_feature_names_out(cat_list)
X_test_ohe=pd.DataFrame(ohe_test,columns=column_name,index=X_test.index)

# bring 'em together
X_train_opt=pd.concat([X_train_scale,X_train_ohe],axis=1)
X_test_opt=pd.concat([X_test_scale,X_test_ohe],axis=1)



# ------------------------------------------------------------

### SAVE FILES for further optimized modeling (directly in GitHub project)    
    
# X_train
filename=r'data\kaggle\Kaggle_X_train_opt.csv'
f=open(filename,'w',encoding='utf-8') # if only write ('w') specified, existing file will be replaced
f.write(X_train_opt.to_csv())
f.close()

# X_test
filename=r'data\kaggle\Kaggle_X_test_opt.csv'
f=open(filename,'w',encoding='utf-8')
f.write(X_test_opt.to_csv())
f.close()

# y_train
filename=r'data\kaggle\Kaggle_y_train_opt.csv'
f=open(filename,'w',encoding='utf-8')
f.write(y_train_opt.to_csv())
f.close()

# y_test
filename=r'data\kaggle\Kaggle_y_test_opt.csv'
f=open(filename,'w',encoding='utf-8')
f.write(y_test_opt.to_csv())
f.close()


# ------------------------------------------------------

## new model
from sklearn.utils.multiclass import type_of_target
type_of_target(y_train_opt)

# instantiate
mnl=LogisticRegression(multi_class='multinomial',random_state=0)
mnl_='Multinomial log'
metrics=["MSE_test","RMSE_test","Train_score","Test_score"]
df_mnl=pd.DataFrame({"model":mnl_,"id":range(6)})

y_train_opt.value_counts()

# GridSearch
start_time=datetime.now()
print("\n Starting the grid search for best mnl model\n at:",start_time.strftime("%H:%M:%S"))

# define grid
grid={'max_iter':range(100,201),
      'solver':["sag"]}
gs_mnl=GridSearchCV(estimator=mnl,param_grid=grid,
                  cv=3,n_jobs=-1,verbose=2)

# train model according to prior defined (hyper)parameters
gs_mnl.fit(X_train_merge,y_train)

# store predictions for later use in evaluation (eg, classification report)
pred_train_mnl=gs_mnl.predict(X_train_opt)
pred_test_mnl=gs_mnl.predict(X_test_opt)

# fill metrics dataframe
df_mnl["MSE test"]=mean_squared_error(y_test_opt,pred_test_mnl)
df_mnl["RMSE test"]=np.sqrt(mean_squared_error(y_test_opt,pred_test_mnl))
df_mnl["Train score"]=gs_mnl.score(X_train_opt,y_train_opt)
df_mnl["Test score"]=gs_mnl.score(X_test_opt,y_test_opt)
df_mnl["Best params"]=str(gs_mnl.best_params_)
elapsed=(datetime.now()-start_time)/timedelta(seconds=1)
df_mnl["Calculation time"]=elapsed
    
# Classification report
print(classification_report(y_test,pred_test_mnl))
    
# How long did it take?
print("Best model parameters found after:",elapsed,"seconds")


# show metrics comparison 
df_mnl
df_mnl.set_index("metric",inplace=True)
df_mnl


# ----------------------------------------------------------------

"""
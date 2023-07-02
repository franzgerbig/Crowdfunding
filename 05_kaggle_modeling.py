# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:21:25 2023

@authors: Hendrik Bosse, Franz Gerbig
"""

# -----------------------------------------------------------------

### Modeling

## settings
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# load train data
X_train_merge=pd.read_csv("data/kaggle/Kaggle_X_train_merge.csv",index_col='id')
X_train_merge.info()
X_test_merge=pd.read_csv("data/kaggle/Kaggle_X_test_merge.csv",index_col='id')
y_train=pd.read_csv("data/kaggle/Kaggle_y_train.csv")
y_train.info()
y_test=pd.read_csv("data/kaggle/Kaggle_y_test.csv")
y_train.value_counts()

# due to computational/RAM limits: reduce data volume
for r in range(2009,2021):
    name="launched_year_"+str(r)
    X_train_merge.drop(name,axis=1,inplace=True)
    X_test_merge.drop(name,axis=1,inplace=True)
for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
    name="launched_day_"+d
    X_train_merge.drop(name,axis=1,inplace=True)
    X_test_merge.drop(name,axis=1,inplace=True)

X_train_merge.shape


## Classification tree
tree=DecisionTreeClassifier()
tree.fit(X_train_merge,y_train)
tree_pred=tree.predict(X_test_merge)
print('Decision Tree R2 value:',tree.score(X_test_merge,y_test)) # Decision Tree R2 value: 0.9969926754118927

## Logistical Regression
log=LogisticRegression()
log.fit(X_train_merge,y_train)
log_pred=log.predict(X_test_merge)
print('Logistical Regression R2 value:',log.score(X_test_merge,y_test)) # Logistical Regression R2 value: 0.9122894411542223

## Random Forest Regression
rfr=RandomForestRegressor()
rfr.fit(X_train_merge,y_train)
rfr_pred=rfr.predict(X_test_merge)
print('Random Forest Classifier R2 value:',rfr.score(X_test_merge,y_test)) # RandomForest Regression R2 value: 0.993898974012259


# Confusion matrix
print('Confusion matrix of Decision Tree')
print(pd.crosstab(y_test,tree_pred,normalize=True, rownames=['True'],colnames=['Predicted']))
print('Confusion matrix of Logistical Regression')
print(pd.crosstab(y_test,log_pred,normalize=True, rownames=['True'],colnames=['Predicted']))
print('Confusion matrix of Random Forest')
print(pd.crosstab(y_test,rfr_pred,normalize=True, rownames=['True'],colnames=['Predicted']))

# Classification Report
from sklearn.metrics import classification_report
print('Classification Report \n\n Decision Tree \n',classification_report(y_test,tree_pred))
print('Logistical Regression \n',classification_report(y_test,log_pred))
print('Random Forest Classifier \n',classification_report(y_test,rfr_pred))



# Buildung the Coefficient DataFrame of the best model so far
log_importance=pd.DataFrame({'Variables':X_train_merge.columns,'Coefficient':log.coef_[0]})
display(log_importance)
log_importance_negatives=log_importance.sort_values('Coefficient',ascending=True)
log_importance_positives=log_importance.sort_values('Coefficient',ascending=False)


# Plotting the Coefficeint DataFrame
plt.figure(figsize=[10,10])
plt.plot(log_importance_positives['Variables'][0:5],log_importance_positives['Coefficient'][0:5])
plt.ylabel('Coefficient')
plt.xlabel('Variables')
plt.title('Top 5 positive coefficients');

plt.figure(figsize=[10,10])
plt.plot(log_importance_negatives['Variables'][0:5],log_importance_negatives['Coefficient'][0:5])
plt.ylabel('Coefficient')
plt.xlabel('Variables')
plt.title('Top 5 negative coefficients');
plt.show()


# Buildung the Coefficient DataFrame of the best model so far
rfc_importance=pd.DataFrame({'Variables':X_train_merge.columns,'Coefficient':log.coef_[0]})
display(rfc_importance)
rfc_importance_negatives=rfc_importance.sort_values('Coefficient',ascending=True)
rfc_importance_positives=rfc_importance.sort_values('Coefficient',ascending=False)


# Plotting the Coefficeint DataFrame
plt.figure(figsize=[10,10])
plt.plot(rfc_importance_positives['Variables'][0:5],rfc_importance_positives['Coefficient'][0:5])
plt.ylabel('Coefficient')
plt.xlabel('Variables')
plt.title('Top 5 positive coefficients');

plt.figure(figsize=[10,10])
plt.plot(rfc_importance_negatives['Variables'][0:5],rfc_importance_negatives['Coefficient'][0:5])
plt.ylabel('Coefficient')
plt.xlabel('Variables')
plt.title('Top 5 negative coefficients');
plt.show()
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def prepare_data(dataset):
    new_df = dataset.copy(deep=True)
    #binarize gender variable
    new_df['gender']=new_df['gender'].replace({new_df['gender'].unique()[0]:0,
                     new_df['gender'].unique()[1]:1})
    #transform target variable
    new_df['match'] =new_df['match'].apply(lambda x :x [2:-1]).astype(int)
    #dropping Nan Columns
    s =(new_df.isna().sum()/new_df.shape[0] > 0.08)
    columns_to_drop =s[s].index.values
    return new_df.select_dtypes(exclude=['object']).drop(columns = columns_to_drop).dropna()

def draw_correlation_with_target(df,nb_features=[40,-1]):
    fig = plt.figure()
    sns.set_theme(style="whitegrid")
    g = sns.barplot(
        x=df[df['gender']==df['gender'].unique()[0]].corr(method="spearman")['match'].sort_values()[nb_features[0]:nb_features[1]],
        y=df[df['gender']==df['gender'].unique()[0]].corr(method="spearman")['match'].sort_values()[nb_features[0]:nb_features[1]].index,
    
    palette="dark", alpha=.6
    )

    g.set(xlabel='Correlation with match', ylabel='Features')
    plt.xticks(rotation=45)
    g.set_title("Relation betweeen some Traits and Desirability (women)")
    return(fig)

options=["KNN","Logistic Regression","Random Forest"]
def get_score(model_name,X_train,X_test,y_train,y_test):
    if model_name==options[0]:
        model = KNeighborsClassifier()
    if model_name==options[1]:
        model = LogisticRegression()
    if model_name==options[2]:
        model = RandomForestClassifier()
    model.fit(X_train,y_train)
    return(model.score(X_test,y_test))

def get_predictions(model_name,user_inputs,X_train,y_train):
    if model_name==options[0]:
        model = KNeighborsClassifier()
    if model_name==options[1]:
        model = LogisticRegression()
    if model_name==options[2]:
        model = RandomForestClassifier()
    model.fit(X_train,y_train)
    return(model.predict(user_inputs))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import crowd_config
import members
import functions

st.set_page_config(
    page_title=crowd_config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)


# Customazation of the Variables, yes =1 and no =0
simp_country=0
simp_currency=0
simp_category=0
over_estimation=0
under_estimation=0
# Attention: Very costly computation 
cat_backers_count=0

# build pages
st.sidebar.title("Menu")
pages=["Intro","Data","Visualization","Preprocessing","Modeling",'Conclusions']
page=st.sidebar.radio("Navigation",options=pages)




# Import and present data
# imp=r"C:\Users\bosse\Desktop\Notebooks\Data\Project\Kaggle_dedub.csv"
imp="Kaggle_deduplicated.csv"
df=pd.read_csv(imp)
df.drop(columns=['Unnamed: 0','id'],inplace=True)
   
    
# fill presentation page
if page==pages[0]:
    st.title(pages[0])
    
    
    st.markdown(f"# {crowd_config.TITLE}")
    st.markdown(f"## {crowd_config.PROMOTION}")
    st.markdown("### Team members")
    for i,member in enumerate(crowd_config.TEAM_MEMBERS):
        st.markdown(member.member_markdown(),unsafe_allow_html=True)
        if i==1:
           st.markdown("---") # add visual separator
    
    st.markdown("### Project Objectives")
    st.markdown("Main objectives of this data analysis project is identify common characteristics of crowdfunding campaigns, and which of those have a positive, and which others a negative relation with a campaign's success.")
    st.sidebar.write('More Information')
    if st.sidebar.checkbox('Continue reading'):    
        st.markdown("### Project description")
        st.markdown('The non-profit and non-governmental organization “Good Life” (in the following also called “NGO”) is thinking about introducing a new campaign (in the following called “project” or “project campaign”, as well) to work on for a limited time. Therefore, the budget for long-term projects must not be used, but rather additional funds gathered. That’s why the NGO wants to run a crowdfunding campaign. To maximize its likelihood of success, Good Life wants to analyze drivers of success in former crowdfunding projects with a machine learning model of data analysis.') 
    

        st.markdown("##### From a technical point of view")
        st.markdown("The analysis should be adaptable easily to future needs of modifications and other analysis projects in different realms.")
        
        st.markdown("##### From a economic point of view")
        st.markdown("Since, the NGO, most of all, is funded by donations, it cannot allocate money to engage external consulting, and rely on less cost-intensive methods, such as a success modeling. Ideally the data to be used for the analysis is freely available instead of causing extra costs.") 
        
        st.markdown("##### From a scientiﬁc point of view")
        st.markdown("Apart from money, Good Life is restricted in time, too. The research concerning the crowdfunding project should be done in the shortest period possible. The goal is not deriving general conclusions, recommendations or proving theories on crowdfunding, but to identify characteristics of a good crowdfunding campaign suited to the NGO’s needs and characteristics.")


# Data
if page==pages[1]:
    st.title(pages[1])    
    
    # present data
    st.write('##### Dataset',df)

    st.sidebar.write('Descriptition')
    rows=df.shape[0]
    cols=df.shape[1]
    
    st.markdown("##### Description")
    st.markdown(f"The dataframe contains {rows} project campaigns (rows) described by {cols} features (columns).")
    if st.sidebar.checkbox('Numerical Variables'):
        df.describe(include="float")
    if st.sidebar.checkbox('Categorical Variables'):
        df.describe(exclude="float")


# Visualization
if page==pages[2]:
    # add title
    st.title(pages[2])
    
    # visual exploration
    st.write('Bar chart')
    st.bar_chart(df['status'])
    
    # countplot
    countplot=sns.countplot(x=df["main_category"],hue=df["status"])
    plt.xlabel(df["main_category"])
    plt.title(f"Frequencies of project categories by status");
    st.pyplot(countplot.fig)    
    
    # interactive
    goal=st.slider('goal_usd')  # 👈 this is a widget
    st.write('For this goal, the number of backers can be described as follows.')
    st.table(df.loc[df.goal_usd==goal].backers_count.describe())

    
# preprocessing
if page==pages[3]:   
    # add title
    st.title(pages[3])


# modeling
if page==pages[4]:   
    # add title
    st.title(pages[4])

    # Checkboxes for Variables
    st.sidebar.write('Options')
    st.sidebar.write('Simplification Options: groups selected variables to reduce modalities')
    if st.sidebar.checkbox('Country (recommended)'):
        simp_country=1
    if st.sidebar.checkbox('Currency'):
        simp_currency=1
    if st.sidebar.checkbox('Category'):
        simp_category=1
    st.sidebar.write('Samplig Options')
    if st.sidebar.checkbox('Oversampling'):
        over_estimator=1
    if st.sidebar.checkbox('Undersampling'):
        under_estimator=1
    # Chosen Variables
    not_used_list=['city', 'usd_pledged','sub_category','backers_count','launched_year','duration']
    num_list=['goal_usd']
    cat_list=['country','currency','main_category','creator_projects']
    
    st.sidebar.write('Add variables')
    if st.sidebar.checkbox('backers_count'):
        num_list.append('backers_count')
    if st.sidebar.checkbox('launched_year'):
        cat_list.append('launched_year')    
    if st.sidebar.checkbox('duration'):
        num_list.append('duration')



    # Variable Simplification
    if cat_backers_count==1:
        df['backers_count'].replace(range(0,101),'0-100',inplace=True)
        df['backers_count'].replace(range(101,1001),'101-1000',inplace=True)
        df['backers_count'].replace(range(1001,10001),'1001-10000',inplace=True)
        df['backers_count'].replace(range(10001,200000),'10000+',inplace=True)
        
    if simp_country==1:
        df['country'].replace(['GB','CA'],'GB,CA',inplace=True)
        df['country'].replace(['DE','FR','IT','ES','NL','SE','DK','CH','IE','BE','AT','NO','LU','PL','GR','SI'],'Europe',inplace=True)
        df['country'].replace(['HK','SG','JP'],'Asia',inplace=True)
        df['country'].replace(['MX'],'South America and Mexico',inplace=True)
        df['country'].replace(['AU','NZ'],'Australia and new Zealand',inplace=True)
        
    if simp_category==1:
        df['main_category'].replace(['music','film & video','games','comics','dance'],'Entertainment',inplace=True)
        df['main_category'].replace(['art','fashion','design','photography','theater'],'Culture',inplace=True)
        df['main_category'].replace(['technology','publishing','food','crafts','journalism'],'Others',inplace=True)
        
    if simp_currency==1:
        df['currency'].replace(['AUD','MXN','SEK','HKD','NZD','DKK','SGD','CHF','JPY','NOK','PLN'],'Others',inplace=True)
     
        
    # Splitting of the data
    # Target Variable
    df['status'].replace(['successful','failed'],[1,0],inplace=True)
    df_y=df['status']
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(df.drop(columns='status',axis=1),df_y,test_size=0.2,random_state=42)
    
    # standardizing of the numerical variables
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    # 
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
        
    # Modelling of the data
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import neighbors
    
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
            
            
    # Models
    # Logistical Regression
    log=LogisticRegression()
    log.fit(X_train_merge,y_train)
    log_pred=log.predict(X_test_merge)
    # Random Forest
    rfc=RandomForestClassifier()
    rfc.fit(X_train_merge,y_train)
    rfc_pred=rfc.predict(X_test_merge)
    # Classification tree
    tree=DecisionTreeClassifier()
    tree.fit(X_train_merge,y_train)
    tree_pred=tree.predict(X_test_merge)
    
    # Results
    
    col1,col2,col3=st.columns(3)
    # Classification Reports
    from sklearn.metrics import classification_report
    rep_log=pd.DataFrame(classification_report(y_test,log_pred,output_dict=True))
    rep_tree=pd.DataFrame(classification_report(y_test,tree_pred,output_dict=True))
    rep_rfc=pd.DataFrame(classification_report(y_test,rfc_pred,output_dict=True))
    # R2 Values
    with col1:
        st.markdown('##### Logistical Regression')
        st.write('R2 Value',log.score(X_test_merge,y_test))
        st.write('Confusion matrix',pd.crosstab(y_test,log_pred,normalize=True, rownames=['True'], colnames=['Prediction']))
        st.write('Classification Report',rep_log)
    with col2:
        st.markdown('##### Random Forest Tree')
        st.write('R2 value',rfc.score(X_test_merge,y_test))
        st.write('Confusion matrix',pd.crosstab(y_test,rfc_pred,normalize=True, rownames=['True'], colnames=['Prediction']))
        st.write('Classification Report',rep_rfc)
    with col3:
        st.markdown('##### Decision Tree')
        st.write('R2 Value',tree.score(X_test_merge,y_test))
        st.write('Confusion matrix', pd.crosstab(y_test,tree_pred,normalize=True, rownames=['True'], colnames=['Prediction']))
        st.write('Classification Report',rep_tree)
    # Estraction of the F1 Scores
    neg_rfc=rep_rfc.iloc[2:3,0:1].to_string().split('score')[1]
    pos_rfc=rep_rfc.iloc[2:3,1:2].to_string().split('score')[1]
    neg_tree=rep_tree.iloc[2:3,0:1].to_string().split('score')[1]
    pos_tree=rep_tree.iloc[2:3,1:2].to_string().split('score')[1]
    neg_log=rep_log.iloc[2:3,0:1].to_string().split('score')[1]
    pos_log=rep_log.iloc[2:3,1:2].to_string().split('score')[1]
    f1_scores=[{'name':'logistical Regression','1':pos_log,'0':neg_log},
               {'name':'Decision Tree','1':pos_tree,'0':neg_tree},
               {'name':'Random Forest Tree','1':pos_rfc,'0':neg_rfc}]
    st.markdown('#### Overview F1-Scores')
    st.write(pd.DataFrame(f1_scores))
    # Visualization, in progress^^
    #st.bar_chart(pd.DataFrame(f1_scores).drop('name',axis=1),x=pd.DataFrame(f1_scores)['name'])
    # mayby plt visualization?
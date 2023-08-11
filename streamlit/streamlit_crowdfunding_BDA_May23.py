# input for terminal (e.g. CMD.exe)
# cd C:\...\GitHub\MAY23_BDA_INT_Crowdfunding\streamlit
# streamlit run streamlit_crowdfunding_BDA_May23.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import crowd_config
import members
from PIL import Image

######
st.set_page_config(
    page_title=crowd_config.TITLE,
    page_icon="images/kickstarter-logo-k-white.png",
)
##### Background
page_img="""
<style>
[data-testid="stAppViewContainer"]{
background-color:#dcf1c5;
background-image: url(https://thumbs.dreamstime.com/z/business-success-3996128.jpg?w=992)}
<style>
"""

# Customazation of the Variables, yes =1 and no =0
simp_country=0
simp_currency=0
over_estimation=0
under_estimation=0
# Attention: Very costly computation 
cat_backers_count=0
# Chosen Variables
not_used_list=['city','main_category']
num_list=['goal_usd','usd_pledged','backers_count','duration','launched_year']
cat_list=['country','currency','creator_projects','sub_category']
# build pages
st.sidebar.title("Menu")
pages=["Introduction","Data Source","Preprocessing I","Data Exploration","Preprocessing II","Modeling",'Results','Conclusions']
page=st.sidebar.radio("Navigation",options=pages)


###### Title
st.markdown(page_img,unsafe_allow_html=True)
if page!=pages[0]:
    st.title('Kickstarter Success Factors')
    
# build function to show table like df.info()
def infoOut(data):
        dfInfo=data.columns.to_frame(name='Column')
        dfInfo['Non-Null Count']=data.notna().sum()
        dfInfo['Dtype']=data.dtypes
        dfInfo.reset_index(drop=True,inplace=True)
        return dfInfo
    
# Checkboxes for Variables
#st.sidebar.write('Modeling Options')
if st.sidebar.checkbox('Modeling Options'):
    st.sidebar.write('Simplification Options')
    if st.sidebar.checkbox('Country (recommended)'):
         simp_country=1
    if st.sidebar.checkbox('Currency'):
         simp_currency=1
    if st.sidebar.checkbox('Category'):
         cat_list.remove('sub_category')
         cat_list.append('main_category')
    st.sidebar.write('Sampling Options')
    if st.sidebar.checkbox('Oversampling'):
        over_estimator=1
    if st.sidebar.checkbox('Undersampling'):
        under_estimator=1
    # Add/delete Variables (if checked variable is deleted from model)
    st.sidebar.markdown('#### Realistic Options (recommended)')
    st.sidebar.write('Info: Deletes Variables for a more realistic model')
    if st.sidebar.checkbox('Number of supporters'):
        num_list.remove('backers_count')
    if st.sidebar.checkbox('Duration of the project'):
        num_list.remove('duration')
    if st.sidebar.checkbox('Launched year'):
        num_list.remove('launched_year')
    if st.sidebar.checkbox('Pledged amount in USD'):
        num_list.remove('usd_pledged')

# Import data (once for all pages)
imp="Kaggle_deduplicated.csv"
imp_org="Kaggle_Dataset.csv"
df=pd.read_csv(imp,index_col='id')
df.drop(columns='Unnamed: 0',inplace=True)
df.sub_category.nunique()

# Error Handling of pyplot visualisations
st.set_option('deprecation.showPyplotGlobalUse', False)

################### Preprocessing and Modelling (available for all pages) #####################################################  

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
if over_estimation==1:
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
        #print('Error',item)
        X_test_merge[item]=0
for item in X_test_merge.columns:
    if item not in X_train_merge.columns:
        #print('Error',item)
        X_train_merge[item]=0
    
#################### Introduction ###############################################################
if page==pages[0]:
    st.title(pages[0])
    st.title('Success Factors on')
    image=Image.open('images\kickstarter-logo-green.png')
    st.image(image)
    st.markdown(f"## {crowd_config.PROMOTION}")
    st.markdown("### Team members")
    for i,member in enumerate(crowd_config.TEAM_MEMBERS):
        st.markdown(member.member_markdown(),unsafe_allow_html=True)
        if i==1:
           st.markdown("---") # add visual separator
    
    st.markdown("### Project Objectives")
    st.markdown("Main objective of this data analysis project is identify")
    st.markdown("- common characteristics of crowdfunding campaigns, and \n - which of those have a positive, and \n - which others a negative relation with a campaign's success.")
    st.markdown("### Project description")
    st.markdown('The non-profit and non-governmental organization “Good Life” (in the following also called “NGO”) is thinking about introducing a new campaign (in the following called “project” or “project campaign”, as well) to work on for a limited time. Therefore, the budget for long-term projects must not be used, but rather additional funds gathered. That’s why the NGO wants to run a crowdfunding campaign. To maximize its likelihood of success, Good Life wants to analyze drivers of success in former crowdfunding projects with a machine learning model of data analysis.') 

    st.markdown("### Project criteria")
    st.markdown("The analysis project should fullfil the following criteria")
    st.markdown("- applicable \n - low cost \n - easily adaptable")

##################### Data Source ################################################################
if page==pages[1]:
    st.title(pages[1])

    # describe data source (web page)
    url="https://www.kaggle.com/yashkantharia/kickstarter-campaigns-dataset-20"
    st.markdown(f'''The original dataset was downloaded from the data science network kaggle and may be accessed (as .csv without any cost) <a href={url} target="_blank" title="go download kaggle raw dataset">here</a>.''',unsafe_allow_html=True)
    
    # present original data
    df_dup=pd.read_csv(imp_org)
    st.write('##### Raw dataset',df_dup.head())
    dups=df_dup.id.duplicated().sum()
    rows=df_dup.shape[0]
    cols=df_dup.shape[1]
    nans=df_dup.isna().any().sum().sum()
    
    # create two page columns
    col1,col2=st.columns(2)
    with col1:
        st.markdown(f"This original dataframe contains {rows} (non-unique) project campaigns (rows, including {dups} duplicates) described by {cols} features (columns), and {nans} missing values:")
    with col2:
        def infoOut(data):
            dfInfo=data.columns.to_frame(name='Column')
            dfInfo['Non-Null Count']=data.notna().sum()
            dfInfo['Dtype']=data.dtypes
            dfInfo.reset_index(drop=True,inplace=True)
            return dfInfo
        # credits: https://stackoverflow.com/questions/64067424/how-to-convert-df-info-into-data-frame-df-info
        st.write(infoOut(df_dup))
    
    # target identification
    st.write("##### Target Variable")
    st.markdown("For predicting success of crowdfunding campaigns the most interesting variable is 'status'. For this, we can also note redundancies: Some projects may not be evaluated, since they belong neither to the category 'successful' nor to 'failed' projects.")

    # countplot
    # status colors by default (tab10): success=#ff7f0e (orange), fail=#1f77b4 (blue)
    # status colors: success=#1f77b4t (blue), fail=#ff7f0e (orange)
    tab10_2a=["#ff7f0e",
             "#1f77b4",
             "#2ca02c",
             "#d62728",
             "#9467bd",
             "#8c564b",
             "#e377c2",
             "#7f7f7f",
             "#bcbd22",
             "#17becf"]
    def countplt_target(data,X):
        sns.set_palette(sns.color_palette(tab10_2a))
        sns.countplot(x=data[X],order=data[X].value_counts().index)
        plt.xticks(rotation=45,ha="right")
        plt.title(f"Frequencies of {X} categories");
    countplot=countplt_target(df_dup,'status')
    st.pyplot(countplot)
    
    # general description
    st.sidebar.write('Description of ...')
    if st.sidebar.checkbox('Numerical Variables'):
        st.write('Summary of numerical variables')
        st.table(df_dup.describe(include=["number"]))
    if st.sidebar.checkbox('Categorical Variables'):
        st.write('Summary of categorical variables')
        st.table(df_dup.describe(exclude=["number"]))
        # st.table(df.select_dtypes("object").value_counts())
    st.markdown("##### Creators")
    st.markdown("Another interesting aspect is, that the 'creator_id' neither is unique. That is: there are creators with more than one project run on Kickstarter.")
        
#################### Preprocsssing I ###########################################################
if page==pages[2]:
    st.title(pages[2])
    st.write("##### Data cleaning")
    st.markdown("In first place, we deleted ...")
    st.markdown("- duplicated rows in terms of the project identificator variable 'id' \n - status categories not relevant for the analysis project")
    st.markdown("Some variables were derived from others ...")
    st.markdown("- year when the kickstarter campaign was launched (launched_year) \n - number of projects a creator has realized on kickstarter (creator_projects)")
    st.markdown("Moreover, the variables of main and sub category to which projects belong, are named vice versa (having more main than sub categories). We reversed that.")

    # deduplicated dataset
    # describe data
    st.sidebar.write('Cleaned data description')
    rows=df.shape[0]
    cols=df.shape[1]
    st.markdown("##### Description of the cleaned dataset")
    st.markdown(f"The dataframe contains {rows} (unique) project campaigns (rows) described by {cols} features (columns).")
        
    # general description
    if st.sidebar.checkbox('Numerical Variables'):
        st.write('Summary of numerical variables')
        st.table(df.describe(include=["number"]))
    if st.sidebar.checkbox('Categorical Variables'):
        st.write('Summary of categorical variables')
        st.table(df.describe(exclude=["number"]))
        # st.table(df.select_dtypes("object").value_counts())
    
    st.write(infoOut(df)) 
        
#################### Data Exploration ###########################################################
if page==pages[3]:
    # add title
    st.title(pages[3])
    
    # divide page into columns
    col1,col2=st.columns(2)
    
    # to maintain color "order" of status categories (orange=success)
    tab10_2b=["#1f77b4",
             "#ff7f0e",
             "#2ca02c",
             "#d62728",
             "#9467bd",
             "#8c564b",
             "#e377c2",
             "#7f7f7f",
             "#bcbd22",
             "#17becf"]
    sns.set_palette(sns.color_palette(tab10_2b))

    # countplot
    def countplotFG(X,HUE):
        plt.figure(figsize=(12,12))
        sns.countplot(x=df[X],hue=df[HUE],order=df[X].value_counts().index)
        plt.xlabel(X)
        plt.xticks(rotation=45,ha="right")
        plt.title(f"Frequencies of {X} by {HUE}")
        plt.legend(labels=["failed","successful"])
    with col1:
        st.pyplot(countplotFG('main_category','status'))  
        st.markdown("- entertainment & art! \n - techies & foodies less welcome")
    
    # status by launched_year 
    def yearplot(data,X,HUE):
        plt.figure(figsize=(12,12))
        sns.countplot(x=data[X],hue=data.replace([1,0],['successful','failed'])[HUE])
        plt.xlabel(X)
        plt.legend(labels=["failed","successful"])
        plt.title(f"Number of projects by years of launch and {HUE}");
    with col2:
        st.pyplot(yearplot(df,"launched_year","status"))
        st.markdown("- What happened 2015? \n - (And 2017?)")

    # number of projects by country    
    countries=pd.read_csv('countries.csv')
    # countries.csv created in countries.py
    def countriesplot():
        countries_sort=countries.sort_values(by="projects_count",ascending=False)
        plt.figure(figsize=(12,12))
        sns.catplot(x="countryname",y="projects_count",kind="bar",data=countries_sort)
        plt.xlabel("Country")
        plt.xticks(rotation=45,ha="right")
        plt.title("Number of projects by country");
    with col1:
        st.pyplot(countriesplot())  
        st.markdown("- Crowdfunding: A US-American phenomenon?")

    # success rate by country
    def countriesplot():
        countries_sort=countries.sort_values(by="success%",ascending=False)
        plt.figure(figsize=(12,12))
        sns.catplot(x="countryname",y="success%",kind="bar",data=countries_sort)
        plt.xlabel("Country")
        plt.xticks(rotation=45,ha="right")
        plt.title("Success rate by country");
    with col2:
        st.pyplot(countriesplot())
        st.markdown("- It's about more than quantity (cf Hong Kong)!")    

################ Preprocessing II ###################################################################
if page==pages[4]:
    st.title(pages[4])
    df_dup=pd.read_csv(imp_org)
    st.markdown('### Variable Selection')
    st.write('Excluding Variables: We exclude some variables on the base of some correlation test. All variable with an non'
            ' impact of the successtate are excluded. The used alpha value was 5 % . We used the Anova Test'
            ' for numerical variables and the chi2 for categorical variables')
    st.write('Variable creation: Derived from the hypotheses that experienced creators are more successful, we created the categorical variable'
            '"creator_projects" which contains informations about the number of projects the creator has done.')
    st.write('Variable Simplification: For some categorical variables, we conducted a simplification in form of grouping to reduce the amount of modalities. The reason for this is that some modalities are very rare and can negatively impact the use ofmaschine learning algorithm')
    st.markdown('##### Original variables')
    col1,col2=st.columns(2)
    with col1:
        st.write('Numerical variables:',df_dup.describe(include=["number"]).columns.to_list())
    with col2:
        st.write('Categorical variables:',df_dup.describe(exclude=["number"]).columns.to_list())
    st.markdown('### Encoding and Scaling')
    st.write('We used the sklearn StandardScaler for the scaling of the numerical variables \n'
            'For the categorical variables we used the one hot ecoder')
    st.markdown('##### Final variables')
    col1,col2=st.columns(2)
    with col1:
        st.write('Numerical variables:',num_list)
    with col2:
        st.write('Categorical variables:',cat_list)
        
################ modeling #######################################################################
if page==pages[5]:   
    # add title
    st.title(pages[5])
    # Modelling of the data
    st.markdown('#### Modeling decision')
    st.write('Realistic option: Given our perspective of a new company tring to identifiy the most important factors for its own campaing we have to exclude some \n'
             'of the most important variables. The reason for this is that these variable contain information we cannot know in our situation. \n'
             'For example the number of supporters, this is the most important variable, but we cannot assess the number of supporters at the beginning of our campaing.')
    st.write('Our Model: For our question we find that chosing the Country simplification and exclude some important variables is the best option giving our research question')
    st.write('Excluded variables: \n' 
            '- Number of supporters \n'
            '- Duration of the project \n'
            '- Launched year \n'
            '- Pledged amount in USD')
    
    st.markdown('#### Model Results')
    st.write('We did some test with over/undersampling and we checked for any overfitting problems, but encountert none.')
    st.write('Due to the computation we decided only to show the result and not include them in an interactive model')
    # image loading
    image=Image.open('images/Results_2.png')
    st.image(image)
    
############# Results ####################################################################
if page==pages[6]:
    st.title(pages[6])
    # Disable error Code
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write('Model selection: The best performing Modell, given our recommended options is the logistical regression')
    
    # Logistical Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression()
    log.fit(X_train_merge,y_train)
    log_pred=log.predict(X_test_merge)
    # Classification Report and Result
    from sklearn.metrics import classification_report
    rep_log=pd.DataFrame(classification_report(y_test,log_pred,output_dict=True))
    st.markdown('##### Logistical Regression')
    col1,col2=st.columns(2)
    with col1:
        st.write('Confusion matrix',pd.crosstab(y_test,log_pred,normalize=True, rownames=['True'], colnames=['Prediction']))
    with col2:
        st.write('Classification Report',rep_log[['0','1']])
    # Coefficients#
    log_importance=pd.DataFrame({'Variables':X_train_merge.columns,'Coefficient':log.coef_[0]})
    log_importance_negatives=log_importance.sort_values('Coefficient',ascending=True)
    log_importance_positives=log_importance.sort_values('Coefficient',ascending=False)
    # Plotting
    st.markdown('##### Important factors')
    col1,col2=st.columns(2)
    def neg_log(X):
        plt.figure(figsize=[5,5])
        plt.plot(X['Variables'][0:5],X['Coefficient'][0:5])
        plt.ylabel('Coefficient')
        plt.xlabel('Variables')
        plt.xticks(rotation=45)
        plt.title('Top 5 negative coefficients');
    with col1:
        st.pyplot(neg_log(log_importance_negatives))
        st.write(log_importance_negatives)
    def pos_log(X):
        plt.figure(figsize=[5,5])
        plt.plot(X['Variables'][0:5],X['Coefficient'][0:5])
        plt.ylabel('Coefficient')
        plt.xlabel('Variables')
        plt.xticks(rotation=45)
        plt.title('Top 5 positive coefficients');
    with col2:
        st.pyplot(pos_log(log_importance_positives))
        st.write(log_importance_positives)
    st.write('Conclusion: The most important factor with a realistic approach is the right category \n'
             ' It seems that more entertainment and Art based categories like Video Games and Photographie tends \n'
             ' to do better. Besides the category the most important factor is an experienced project creator and \n'
             ' a an avoidence for too ambitious goals.')
    
############# Conclusion ####################################################################
if page==pages[7]:
    st.title(pages[7])
    st.markdown("### Summary")
    st.markdown("We can sum up the results in two recommendations")
    st.markdown("- start small, splitting your project/idea in multiple campaigns with lower goals \n- by this: gain experience in campaign running and trust by (potential) backers \n- Choose your category wisely.")
    st.markdown("### Projects criteria")
    st.markdown("The analysis model is")
    st.markdown("- applicable to real life \n - of low cost (freely available data & coding software), and \n - easily adaptable (since it's written in code)")
    st.markdown("### Further analyses")    
    st.markdown("Further analyses might build upon the results focusing on other target(s) such as the number of backers and amount of money they pledge to support a project with.")


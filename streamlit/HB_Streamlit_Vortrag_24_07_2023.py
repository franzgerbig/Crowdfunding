import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# import FG_functions_19_07_2023
# cd C:\Users\Franz.000\Documents\GitHub\MAY23_BDA_INT_Crowdfunding\streamlit
# streamlit run HB_Streamlit_Vortrag_26_07_2023.py

# Coloring
page_img="""
<style>
[data-testid="stAppViewContainer"]{
background-color:#dcf1c5;
background-image: url(https://thumbs.dreamstime.com/z/business-success-3996128.jpg?w=992)}
<style>
"""
st.markdown(page_img,unsafe_allow_html=True)
st.title("Kickstarter Success Factors")

# build member class
class Member:
    def __init__(
        self, name: str, linkedin_url: str = None, github_url: str = None
    ) -> None:
        self.name = name
        self.linkedin_url = linkedin_url
        self.github_url = github_url
    def member_markdown(self):
        markdown = f'<b style="display: inline-block; vertical-align: middle; height: 100%">{self.name}</b>'
        if self.linkedin_url is not None:
            markdown += f' <a href={self.linkedin_url} target="_blank"><img src="https://dst-studio-template.s3.eu-west-3.amazonaws.com/linkedin-logo-black.png" alt="linkedin" width="25" style="vertical-align: middle; margin-left: 5px"/></a> '
        if self.github_url is not None:
            markdown += f' <a href={self.github_url} target="_blank"><img src="https://dst-studio-template.s3.eu-west-3.amazonaws.com/github-logo.png" alt="github" width="20" style="vertical-align: middle; margin-left: 5px"/></a> '
        return markdown

# assign content to each member
TEAM_MEMBERS = [
    Member(
        name="Franz Gerbig",
        linkedin_url="https://www.linkedin.com/in/franzgerbig",
        github_url="https://github.com/franzgerbig"
    ),
    Member( 
        name="Hendrik Bosse",
        github_url="https://github.com/hebosse",
    )
]


# prevent future deprecation warning in visualizations
st.set_option('deprecation.showPyplotGlobalUse', False)


# Customazation of the Variables, yes =1 and no =0
simp_country=0
simp_currency=0
over_estimation=0
under_estimation=0
# Attention: Very costly computation 
cat_backers_count=0
# Chosen Variables
not_used_list=['city','main_category']
num_list=['goal_usd','usd_pledged','backers_count','duration']
cat_list=['country','currency','creator_projects','sub_category','launched_year']
# build pages
st.sidebar.title("Menu")
pages=["Introduction","Data Source","Data Exploration","Preprocessing","Modeling",'Results','Conclusions']
page=st.sidebar.radio("Navigation",options=pages)

# Checkboxes for Variables
st.sidebar.write('Options')
if st.sidebar.checkbox('Show options'):
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
    # Add Variables
    st.sidebar.markdown('#### Realistic Options (recommended)')
    st.sidebar.write('Info: Deletes Variables for a more realistic model')
    if st.sidebar.checkbox('Number of supporters'):
        num_list.remove('backers_count')
    if st.sidebar.checkbox('Duration of the project'):
        num_list.remove('duration')
    if st.sidebar.checkbox('Launched year'):
        cat_list.remove('launched_year')
    if st.sidebar.checkbox('Pledged amount in USD'):
        num_list.remove('usd_pledged')
        
# Import and present data
# imp=r"C:\Users\bosse\Desktop\Notebooks\Data\Project\Kaggle_dedub.csv"
imp="Kaggle_deduplicated.csv"
df=pd.read_csv(imp)

# minimal data cleaning
df=df[(df['status']=='successful')|(df['status']=='failed')]
df.drop_duplicates(keep='first',inplace=True,subset='id')
df.drop("Unnamed: 0",axis=1,inplace=True)

# reverse sub and main category column names
new={"main_category":"sub_category2",
"sub_category":"main_category"}
df.rename(columns=new,inplace=True)
df.rename(columns={"sub_category2":"sub_category"},inplace=True)

    
################### Calculations and Modelling #####################################################  

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
    
#################### Introduction ###############################################################
if page==pages[0]:
    st.title(pages[0])
    # st.markdown(f"# {crowd_config.TITLE}")
    # st.markdown(f"## {crowd_config.PROMOTION}")

    st.markdown(f"## Bootcamp Data Scientist\n ### May 2023")
    st.markdown("### Team members")
    for i,member in enumerate(TEAM_MEMBERS):
        st.markdown(member.member_markdown(),unsafe_allow_html=True)
        if i==1:
           st.markdown("---") # add visual separator
    
    st.markdown("### Project Objectives")
    st.markdown("Main objectives of this data analysis project is identify common characteristics of crowdfunding campaigns, and which of those have a positive, and which others a negative relation with a campaign's success.")
    st.markdown("### Project description")
    st.markdown('The non-profit and non-governmental organization “Good Life” (in the following also called “NGO”) is thinking about introducing a new campaign (in the following called “project” or “project campaign”, as well) to work on for a limited time. Therefore, the budget for long-term projects must not be used, but rather additional funds gathered. That’s why the NGO wants to run a crowdfunding campaign. To maximize its likelihood of success, Good Life wants to analyze drivers of success in former crowdfunding projects with a machine learning model of data analysis.') 

    st.markdown("##### From a technical point of view")
    st.markdown("The analysis should be adaptable easily to future needs of modifications and other analysis projects in different realms.")
    
    st.markdown("##### From a economic point of view")
    st.markdown("Since, the NGO, most of all, is funded by donations, it cannot allocate money to engage external consulting, and rely on less cost-intensive methods, such as a success modeling. Ideally the data to be used for the analysis is freely available instead of causing extra costs.") 
    
    st.markdown("##### From a scientiﬁc point of view")
    st.markdown("Apart from money, Good Life is restricted in time, too. The research concerning the crowdfunding project should be done in the shortest period possible. The goal is not deriving general conclusions, recommendations or proving theories on crowdfunding, but to identify characteristics of a good crowdfunding campaign suited to the NGO’s needs and characteristics.")



##################### Data Source ################################################################
if page==pages[1]:
    st.title(pages[1])   

    # some words about data source, cleaning and so on    
    st.markdown("##### Data Source")
    df_dup=pd.read_csv("Kaggle_Dataset.csv")
    dups=df_dup.id.duplicated().sum()
    st.markdown("In first place, we deleted ...")
    st.markdown(f"- {dups} duplicated rows in terms of the project identificator variable 'id'\n - status categories not relevant for the analysis project")
    st.markdown("Moreover, some variables were created ...")
    st.markdown("- year when the kickstarter campaign was launched (launched_year) \n - number of projects a creator has realized on kickstarter (creator_projects)")

    # present data
    st.write('##### Dataset',df.head())
    st.markdown("The resulting dataset is the following")
    if st.button("Show complete dataset"):
        st.write('Complete dataset',df)
    
    # describe data
    rows=df.shape[0]
    cols=df.shape[1]
    st.markdown("##### Description of the used dataset")
    st.markdown(f"The dataframe contains {rows} (unique) project campaigns (rows) described by {cols} features (columns).")
    
    # reverse column names of category variables correctly
    new={"main_category":"sub_category2",
    "sub_category":"main_category"}
    df_dup.rename(columns=new,inplace=True)
    df_dup.drop("sub_category2",axis=1,inplace=True)
    
    # general description
    numeric=list(df_dup.select_dtypes("number").columns)
    # numeric.remove(["id","creator_id"])
    categoric=list(df_dup.select_dtypes("object").columns)
    # categoric.append(["id","creator_id"])
    st.sidebar.write('Description')
    if st.sidebar.checkbox('Numerical Variables'):
        st.table(df_dup.describe(include=numeric))
    if st.sidebar.checkbox('Categorical Variables'):
        st.table(df_dup.describe(include=categoric))


    st.write("##### Data cleaning")    
    
    # target identification
    st.write("#### Target Variable")
    st.markdown("Since we want to predict the success of crowdfunding campaigns on Kickstarter, the most interesting variable is 'status' - the target variable. For this, we can also note redundancies: Some projects may not be evaluated, since they belong neither to the category 'successful' nor to 'failed' projects. Those (unclear) projects will be dropped.")
    # plot the categories
    def countplt_status():
        sns.countplot(data=df_dup,x="status")
        plt.title("Frequencies of status categories");
    st.pyplot(countplt_status())
    
    if st.button("Clean target"):
        # Only Successful and failed projects are important for us
        df_dup=df_dup.loc[(df['status']=='successful')|(df['status']=='failed')]

    # duplicates (id)
    st.write("#### Duplicates (?)")
    st.markdown("Do we need to account for duplicated rows in the dataset? Let's check this in terms of the variable 'id'.")    
    if st.button("Check duplicates",help="Click to check for and delete duplicates (if applicable)."):
        # general_dups=df_dup.duplicated.sum().sum()
        id_dups=df_dup.id.duplicated.sum()
        if id_dups>0:
            df_dup=drop_duplicates(keep='first',inplace=True,subset='id')
            st.markdown("Congratulations - you deleted {id_dups} duplicated rows!")
        else: 
            st.markdown("Relax - no need to delete duplicated rows.")
    
    # creator (creator_id)
    st.write("##### Creators")
    st.markdown("Another interesting aspect is, that the creator_id neither is unique. That is: there are creators with more than one projects run on Kickstarter. We will come back to this later ...")


#################### Data Exploration ###########################################################
if page==pages[2]:
    # add title
    st.title(pages[2])
    
    # reverse main and sub category
    new={"main_category":"sub_category2",
    "sub_category":"main_category"}
    df.rename(columns=new,inplace=True)
    df.drop("sub_category2",axis=1,inplace=True)
    
    # countplot main category by status
    def countplotFG(X,HUE):
        sns.countplot(x=df[X],hue=df[HUE])
        plt.xlabel(X)
        plt.xticks(rotation=45)
        plt.title(f"Frequencies of {X} by {HUE}");
    countplot=countplotFG('main_category','status')
    st.pyplot(countplot)      
    
    # (world) map
    countries=pd.read_csv("countries.csv")
    st.map(data=countries,size="projects_count", zoom=None,use_container_width=True)


################ preprocessing ###################################################################

if page==pages[3]:
    st.title(pages[3])
    st.markdown('### Variable Selection')
    st.write('Target Variable: For our target variable we chose the "status", the varibale contains the information about the projects sucess'
         'given the project question, we focus only on the projects which are either successful or have failed')
    st.write('Excluding Variables: We exclude some variables on the base of some correlation test. All variable with an non'
            ' impact of the successtate are excluded. The used alpha value was 5 % . We used the Anova Test'
            ' for numerical variables and the chi2 for categorical variables')
    st.write('Variable creation: Derived from the hypotheses that experienced creators are more successful, we created the categorical variable'
            '"creator_projects" which contains informations about the number of projects the creator has done')
    col1,col2=st.columns(2)
    with col1:
        st.write('Numerical variables:',num_list)
    with col2:
        st.write('Categorical variables:',cat_list)
    
    st.markdown('#### Encoding and Scaling')
    st.write('We used the sklearn StandardScaler for the scaling of the numerical variables \n'
            'For the categorical variables we used the one hot ecoder')
    st.write('preprcessed training dataset', X_train_merge[:100])
    
################ modeling #######################################################################
if page==pages[4]:   
    # add title
    st.title(pages[4])
    # Modelling of the data
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import neighbors
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
    st.write('Model selection: The used Modell is the logistical regression with an simplification of the country variable for an increased prediction value \n'
        'and an exclusion of the variables backers_count, duration, launched_year, and usd_pledged')

############# Results ####################################################################
if page==pages[5]:
    st.title(pages[5])
    
    # Disable error Code
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Logistical Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression()
    log.fit(X_train_merge,y_train)
    log_pred=log.predict(X_test_merge)
    # Coefficients
    log_importance=pd.DataFrame({'Variables':X_train_merge.columns,'Coefficient':log.coef_[0]})
    log_importance_negatives=log_importance.sort_values('Coefficient',ascending=True)
    log_importance_positives=log_importance.sort_values('Coefficient',ascending=False)
    # Plotting
    col1,col2=st.columns(2)
    def neg_log(X):
        plt.figure(figsize=[5,5.5])
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
if page==pages[6]:
    st.title(pages[6])
    # te='heyy'
    # st.markdown(f"Platzhalter :blue[{te}]")
    
    st.markdown("#### Recommendations")
    st.markdown("We can sum up the results in two recommendations")
    st.markdown("- start small, splitting your project/idea in multiple campaigns with lower goals \n- by this: gain experience in campaign running and trust by (potential) backers \n - Choose your category wisely.")
    st.markdown("Further analyses might build upon the results focusing on other target(s) such as the number of backers and amount of money they pledge to support a project with.")
    st.markdown("#### Model apllication")
    st.markdown("The model can be useful in determining the financial strategy at the beginning of a project. Given the highly important influence of the category, choosing a Kickstarter crowdfunding as a financing strategy is far more viable for entertainment-based projects. \nTherefore, the NGO should take good care of by which medium and in which context, it wants to realize its campaign and the message it wants to spread out with it. \n\nIf the entertainment-related type of campaign is not possible should choose a different financial strategy. The other conclusion is more viable as a general rule, don’t shoot for the moon, build up slowly and carefully. Following such a strategy of lower funding goals implies a longer time until the (final) overall goal can be reached.")
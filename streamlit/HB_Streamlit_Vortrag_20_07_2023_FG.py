import streamlit as st
import pandas as pd
import numpy as np
import crowd_config
import members
import functions
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# (main) title
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
# Chosen Variables
not_used_list=['city', 'usd_pledged','sub_category','backers_count','launched_year','duration']
num_list=['goal_usd']
cat_list=['country','currency','main_category','creator_projects']
# build pages
st.sidebar.title("Menu")
pages=["Introduction","Data Source","Data Exploration","Preprocessing","Modeling",'Results','Conclusions']
page=st.sidebar.radio("Navigation",options=pages)

if page==pages[4] or page==pages[5]:
    # Checkboxes for Variables
    st.sidebar.write('Simplification Options')
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
    # Add Variables
    st.sidebar.write('Add variables')
    if st.sidebar.checkbox('backers_count (grouped, not recommended)'):
        num_list.append('backers_count')
    if st.sidebar.checkbox('duration'):
        num_list.append('duration')
    if st.sidebar.checkbox('launched_year'):
        cat_list.append('launched_year')



# Import data
# imp=r"C:\Users\bosse\Desktop\Notebooks\Data\Project\Kaggle_dedub.csv"
imp=r"C:\Users\bosse\Desktop\Notebooks\Data\Project\Kaggle_Dataset.csv"
imp="Kaggle_Dataset.csv"
df=pd.read_csv(imp)

# save duplicated dataset
df_dup=df

# minimal data cleaning
df=df[(df['status']=='successful')|(df['status']=='failed')]
df.drop_duplicates(keep='first',inplace=True,subset='id')
df.drop("Unnamed: 0",axis=1,inplace=True)

# make preprocessing available on all relevant pages
if (page==pages[3] or page==pages[4] or page==pages[5]):
    from functions import preprocess
    preprocess()
    
# make modeling available on all relevant pages
if (page==pages[4] or page==pages[5]):
    from functions import preprocess
    modeling()

   
################### Calculations and Modelling #####################################################  

# Variable Simplification
if cat_backers_count==1:
    simple_backers()
        
if simp_country==1:
    simple_country()
        
if simp_category==1:
    simple_category()
        
if simp_currency==1:
    simple_currency()

     
        


    
#################### Introduction ###############################################################
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
    

    # present data
    st.write('##### Dataset',df_dup.head())

    # describe data
    st.sidebar.write('Description')
    rows=df_dup.shape[0]
    cols=df_dup.shape[1]
    st.markdown("##### Description")
    st.markdown(f"The dataframe contains {rows} (unique) project campaigns (rows) described by {cols} features (columns).")
    
    # reverse column names of category variables correctly
    new={"main_category":"sub_category2",
    "sub_category":"main_category"}
    df_dup.rename(columns=new,inplace=True)
    df_dup.drop("sub_category2",axis=1,inplace=True)
    
    # general description
    if st.sidebar.checkbox('Numerical Variables'):
        st.table(df_dup.describe(include=["number"]))
    if st.sidebar.checkbox('Categorical Variables'):
        st.table(df_dup.describe(exclude=["number"]))

    st.write("##### Data cleaning")
    st.markdown("If you pay attention to the description of the variable 'id', you'll notice, it's not unique, which is, some projects appear more than once. For all further steps, we will keep only the first ocurrence of each project id, respectively.")
    st.markdown("Another interesting aspect is, that the creator_id neither is unique. That is: there are creators with more than one projects run on Kickstarter. We will come back to this later.")
    
    # target identification
    st.write("##### Target Variable")
    st.markdown("Since, we want to predict the success of crowdfunding campaigns on Kickstarter, the most interesting variable is 'status' - the target variable. For this, we can also note redundancies: Some projects may not be evaluated, since they belong neither to the category 'successful' nor to 'failed' projects. Those (unclear) projects will be dropped, too.")
        
        
#################### Data Exploration ###########################################################
if page==pages[2]:
    # add title
    st.title(pages[2])
    
    # visual exploration
    #st.write('Bar chart')
    #st.bar_chart(df['status'])
    
    # reverse main and sub category
    new={"main_category":"sub_category2",
    "sub_category":"main_category"}
    df.rename(columns=new,inplace=True)
    df.drop("sub_category2",axis=1,inplace=True)
    
    # countplot
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
    
    # subplots
    
    
    # interactive
    # for c in df.main_category:
        # if st.selectbox(df.main_category.value_counts().sort_index(ascending=True)())==c:
            
    

############### preprocessing ###################################################################

if page==pages[3]:
    st.title(pages[3])


################# modeling #######################################################################
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


############# Results ####################################################################
if page==pages[5]:
    st.title(pages[5])
    from sklearn.linear_model import LogisticRegression
    # Logistical Regression
    log=LogisticRegression()
    log.fit(X_train_merge,y_train)
    log_pred=log.predict(X_test_merge)
    # Coefficients
    log_importance=pd.DataFrame({'Variables':X_train_merge.columns,'Coefficient':log.coef_[0]})
    log_importance_negatives=log_importance.sort_values('Coefficient',ascending=True)
    log_importance_positives=log_importance.sort_values('Coefficient',ascending=False)
    # Plotting
    def neg_log(X):
        plt.figure(figsize=[7,7])
        plt.plot(X['Variables'][0:5],X['Coefficient'][0:5])
        plt.ylabel('Coefficient')
        plt.xlabel('Variables')
        plt.xticks(rotation=45)
        plt.title('Top 5 negative coefficients');
    st.pyplot(neg_log(log_importance_negatives))
    
    def pos_log(X):
        plt.figure(figsize=[7,7])
        plt.plot(X['Variables'][0:5],X['Coefficient'][0:5])
        plt.ylabel('Coefficient')
        plt.xlabel('Variables')
        plt.xticks(rotation=45)
        plt.title('Top 5 positive coefficients');
    st.pyplot(pos_log(log_importance_positives))
    
    
############# Conclusion ####################################################################
if page==pages[6]:
    st.title(pages[6])
    
    st.markdown("We can sum up the results in two recommendations")
    st.markdown("- start small, splitting your project/idea in multiple campaigns with lower goals \n- by this: gain experience in campaign running and trust by (potential) backers \n-Choose your category wisely.")
    st.markdown("Further analyses might build upon the results focusing on other target(s) such as the number of backers and amount of money they pledge to support a project with.")
    st.markdown("The model can be useful in determining the financial strategy at the beginning of a project. Given the highly important influence of the category, choosing a Kickstarter crowdfunding as a financing strategy is far more viable for entertainment-based projects. \nTherefore, the NGO should take good care of by which medium and in which context, it wants to realize its campaign and the message it wants to spread out with it. \n\nIf the entertainment-related type of campaign is not possible should choose a different financial strategy. The other conclusion is more viable as a general rule, don’t shoot for the moon, build up slowly and carefully. Following such a strategy of lower funding goals implies a longer time until the (final) overall goal can be reached.")
# cd C:\Users\Franz.000\Documents\GitHub\MAY23_BDA_INT_Crowdfunding\streamlit
# streamlit run HB_Streamlit_Vortrag_26_07_2023.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import scipy.stats as stats
# import Franz asugelagerter Code
import crowd_config
import members


# Coloring
# background-image: url(https://thumbs.dreamstime.com/z/business-success-3996128.jpg?w=992)}
[theme]
primaryColor="#F63366"
backgroundColor="#05ce78"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

page_img="""
<style>
[data-testid="stAppViewContainer"]{
background-color:#dcf1c5;
background-image: url(https://thumbs.dreamstime.com/z/business-success-3996128.jpg?w=992)}
<style>
"""

###### Title
st.markdown(page_img,unsafe_allow_html=True)
st.title('Kickstarter Success Factors')

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
imp=r"C:\Users\bosse\Desktop\Notebooks\Data\Project\Kaggle_dedub.csv"
imp_org=r"C:\Users\bosse\Desktop\Notebooks\Data\Project\Kaggle_Dataset.csv"
imp="Kaggle_deduplicated.csv"
imp_org="Kaggle_Dataset.csv"
df=pd.read_csv(imp,index_col='id')
df.drop(columns='Unnamed: 0',inplace=True)

# reverse column names of category variables correctly
new={"main_category":"sub_category2",
"sub_category":"main_category"}
df.rename(columns=new,inplace=True)
df.rename(columns={"sub_category2":"sub_category"},inplace=True)

# Error Handling of pyplot visualisations
st.set_option('deprecation.showPyplotGlobalUse', False)

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
        #print('Error',item)
        X_test_merge[item]=0
for item in X_test_merge.columns:
    if item not in X_train_merge.columns:
        #print('Error',item)
        X_train_merge[item]=0
    
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
    st.markdown("Main objective of this data analysis project is identify")
    st.markdown("- common characteristics of crowdfunding campaigns, and \n - which of those have a positive, and \n - which others a negative relation with a campaign's success.")
    st.markdown("### Project description")
    st.markdown('The non-profit and non-governmental organization “Good Life” (in the following also called “NGO”) is thinking about introducing a new campaign (in the following called “project” or “project campaign”, as well) to work on for a limited time. Therefore, the budget for long-term projects must not be used, but rather additional funds gathered. That’s why the NGO wants to run a crowdfunding campaign. To maximize its likelihood of success, Good Life wants to analyze drivers of success in former crowdfunding projects with a machine learning model of data analysis.') 

    st.markdown("### Project criteria")
    st.markdown("The analysis project should fullfil the following criteria")
    st.markdown("- applicable \n - low cost \n - easily adaptable")
    
    # st.markdown("##### From a technical point of view")
    # st.markdown("The analysis should be adaptable easily to future needs of modifications and other analysis projects in different realms.")
    
    # st.markdown("##### From a economic point of view")
    # st.markdown("Since, the NGO, most of all, is funded by donations, it cannot allocate money to engage external consulting, and relies on less cost-intensive methods, such as a success modeling. Ideally the data to be used for the analysis is freely available instead of causing extra costs.") 
    
    # st.markdown("##### From a scientiﬁc point of view")
    # st.markdown("The goal is not deriving general conclusions, recommendations or proving theories on crowdfunding, but to identify characteristics of a good crowdfunding campaign suited to the NGO’s needs and characteristics.")

##################### Data Source ################################################################
if page==pages[1]:
    st.title(pages[1])

    # describe data source (web page)
    url="https://www.kaggle.com/yashkantharia/kickstarter-campaigns-dataset-20"
    st.markdown(f'''The original dataset was downloaded from the data science network kaggle and may be accessed (as .csv without any cost) <a href={url} target="_blank" title="go download kaggle raw dataset">here</a>. ''',unsafe_allow_html=True)

    #
    st.table(df.info())
    
    # present original data
    df_dup=pd.read_csv(imp_org)
    st.write('##### Raw dataset',df_dup.head())
    rows=df_dup.shape[0]
    cols=df_dup.shape[1]
    st.markdown(f"This original dataframe contains {rows} (non-unique) project campaigns (rows) described by {cols} features (columns).")

    st.write("##### Data cleaning")
    dups=df_dup.id.duplicated().sum()
    st.markdown("In first place, we deleted ...")
    st.markdown(f"- {dups} duplicated rows in terms of the project identificator variable 'id' \n - status categories not relevant for the analysis project")
    st.markdown("Some variables were created ...")
    st.markdown("- year when the kickstarter campaign was launched (launched_year) \n - number of projects a creator has realized on kickstarter (creator_projects)")
    st.markdown("Moreover, the variables of main and sub category are named vice versa (having more value). We reversed that back.")

    st.markdown("If you pay attention to the description of the variable 'id', you'll notice, it's not unique, which is, some projects appear more than once. For all further steps, we will keep only the first ocurrence of each project id, respectively.")
    
    # turn to deduplicated dataset
    # describe data
    st.sidebar.write('Data description')
    rows=df.shape[0]
    cols=df.shape[1]
    st.markdown("##### Description")
    st.markdown(f"The dataframe contains {rows} (unique) project campaigns (rows) described by {cols} features (columns).")
        
    # general description
    if st.sidebar.checkbox('Numerical Variables'):
        st.write('Summary of numerical variables')
        st.table(df.describe(include=["number"]))
    if st.sidebar.checkbox('Categorical Variables'):
        st.write('Summary of categorical variables')
        st.table(df.describe(exclude=["number"]))
        # st.table(df.select_dtypes("object").value_counts())
    
        
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
    
    # countplot
    def countplotFG(X,HUE):
        sns.countplot(x=df[X],hue=df[HUE],order=df[X].value_counts().index)
        plt.xlabel(X)
        plt.xticks(rotation=45,ha="right")
        plt.title(f"Frequencies of {X} by {HUE}");
    countplot=countplotFG('main_category','status')
    st.pyplot(countplot)  
    st.markdown("- some explanation summary \n - some more")
    
    # interactive
    # nr_backers=st.slider('Quantiles Number of backers',value=100)
    # quantiles=df['backers_count'].quantile(q=[0,nr_backers/100])
    # st.write(quantiles.to_list()[1])
    # def boxplot(x):
    #     sns.boxplot(x)
    #     plt.xlabel('Number of backers')
    #     plt.title('Distribution of the number of backers per project.')
    # df_backers=df.loc[(df['backers_count']<=quantiles.to_list()[1])]
    # st.pyplot(boxplot(df_backers['backers_count']))
    
    # status by launched_year 
    def yearplot(data,X,HUE):
        plt.figure(figsize=(7,7))
        sns.countplot(x=data[X],hue=data[HUE])
        plt.xlabel(X)
        plt.xticks(rotation=45)
        plt.title(f"Number of projects by years of launch and {HUE}");
    st.pyplot(yearplot(df,"launched_year","status"))
    
    # goal=st.slider('Quantiles Number of backers',value=100)
    # quantiles=df['goal_usd'].quantile(q=[0,goal/100])
    # st.write(quantiles.to_list()[1])
    # def boxplot(x):
    #     sns.boxplot(x)
    #     plt.xlabel('Funding goal in USD')
    #     plt.title('Distribution of the goal amount per project.')
    # df_goal=df.loc[(df['goal_usd']<=quantiles.to_list()[1])]
    # st.pyplot(boxplot(df_goal['goal_usd']))

    # success rate by country
    def violinplot():
        plt.figure(figsize=(7,7))
        sns.catplot(x="countryname",y="success%",kind="violin",data=countries)
        plt.xlabel("Country")
        plt.xticks(rotation=45,ha="right")
        plt.title("Success rate by country");
    st.pyplot(violinplot())
    
    # (world) map
    countries=pd.read_csv('countries.csv')
    # st.map(data=countries,latitude="lat",longitude="lon",size=df["projects_count"],zoom=None,use_container_width=True)
    
    # with mapbox (pdk)
    # st.pydeck_chart(pdk.Deck(
    #     map_style=None,
    #     initial_view_state=pdk.ViewState(
    #         latitude=37.76,
    #         longitude=-122.4,
    #         zoom=11,
    #         pitch=50,
    #     ),
    #     layers=[
    #         pdk.Layer(
    #            'HexagonLayer',
    #            data=countries,
    #            get_position='[lon, lat]',
    #            radius=200,
    #            elevation_scale=4,
    #            elevation_range=[0, 1000],
    #            pickable=True,
    #            extruded=True,
    #         ),
    #         pdk.Layer(
    #             'ScatterplotLayer',
    #             data=countries,
    #             get_position='[lon, lat]',
    #             get_color='[200, 30, 0, 160]',
    #             get_radius=200,
    #         ),
    #     ],
    #     )
    # )
    
    # instead of signing up for the Mapbox service, we can use OpenStreetMap, which is free to use and works directly with Plotly
    # fig=px.scatter_mapbox(countries,lat="lat",lon="lon",zoom=None,size=df["projects_count"])
    # fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # st.plotly_chart(fig)

################ preprocessing ###################################################################

if page==pages[3]:
    st.title(pages[3])
    st.markdown('### Variable Selection')
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
if page==pages[6]:
    st.title(pages[6])
    st.markdown("### Summary")
    st.markdown("We can sum up the results in two recommendations")
    st.markdown("- start small, splitting your project/idea in multiple campaigns with lower goals \n- by this: gain experience in campaign running and trust by (potential) backers \n- Choose your category wisely.")
    st.markdown("### Projects criteria")
    st.markdown("The analysis model is")
    st.markdown("- applicable to real life \n - of low cost (freely available data & coding software), and \n - easily adaptable (since it's written in code)")
    # st.markdown('The model can be useful in determining the financial strategy at the beginning of a project. Given the highly important influence of the category, choosing a Kickstarter crowdfunding as a financing strategy is far more viable for entertainment-based projects. \n'
    # 'Therefore, the NGO should take good care of by which medium and in which context, it wants to realize its campaign and the message it wants to spread out with it. \n\nIf the entertainment-related type of campaign is not possible a different financial strategy should be chosen. \n'
    # 'Following a strategy of lower funding goals implies a longer time until the (final) overall goal can be reached.')
    
    st.markdown("### Further analyses")    
    st.markdown("Further analyses might build upon the results focusing on other target(s) such as the number of backers and amount of money they pledge to support a project with.")


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd 
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from util import prepare_data,draw_correlation_with_target, get_score ,get_predictions
st.sidebar.title("Menu")
pages =["Presentation","Visualization","Modeling", 'Conclusions']
page = st.sidebar.radio("Choose a page",options = pages)
df = pd.read_csv("Kaggle_Dataset.csv")
df_final = prepare_data(df)
features_list = ["gender","age","age_o","attractive_o","sinsere_o","funny_o","intelligence_o",
     "funny_partner","attractive_partner","sincere_partner","intelligence_partner"]
X_train,X_test,y_train, y_test = train_test_split(df_clean [
    features_list ],df_clean['match'])


if page == pages[0]: 
    st.title('Introduction to streamlit')
    st.header("By Datascientest")
    st.image("dataset-cover.jpg",caption="A cool picture about the dataset")
    st.video("https://www.youtube.com/watch?v=WKNRM2xVRJo")
    st.write("The columns' names are")
    st.markdown("Here is the [link](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment) towards the original dataset")
    st.dataframe(df_clean.head())
    st.markdown("Here is the [link](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment) towards the original dataset")
    st.latex(r'''
    p = \dfrac{1}{1+e^{-{b_{0} + b_{1} \times x }}}
    ''')

if page  == pages[1]: 
    st.title('Visualization')
    st.header("")

    # fig,ax = plt.subplots()

    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df_clean, kind="bar",
        x="gender", y="funny", hue="match",
    palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("Gender", "Funny ")
    plt.xticks(rotation=45)
    g.legend.set_title("Match ? ")

    st.pyplot(g)

    fig,ax = plt.subplots()
    sns.boxplot(x="gender", y="funny_o",
            hue="match", palette=["m", "g"],
            data=df_clean)
    sns.despine(offset=10, trim=True)
    st.pyplot(fig)
    

    fig3 = plt.figure()
    sns.displot(
    df, x="age", col="race", row="gender",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True),
    )
    st.pyplot(fig3)
    number_start = st.number_input('Insert a number for first feature ',0,len(df_clean.columns)-1,1)
    number_end= st.slider('Number of features?',min_value= int(number_start), max_value=int(len(df_clean.columns)) ,step= 1)
    st.write('The current number is ', number_start)
    st.write('The current number is ', number_end)
    if number_start < number_end:
        st.pyplot(draw_correlation_with_target(df_clean,nb_features=[number_start,number_end]))
    
if page == pages[2]:
    model_name =st.selectbox("Choose a ML Model to train",options=["KNN","Logistic Regression","Random Forest"])
    st.write(f"The performance of the ML model is {get_score(model_name,X_train,X_test,y_train,y_test)}")

    features_range =[list(df_clean[f].unique()) for f in features_list]
    options ={}
    for i in range(len(features_list))  : 
        options[features_list[i]]= st.selectbox(
        f'What is your {features_list[i]} ?',
      features_range[i]
        )

    st.write('You selected:', options)

    inputs = pd.DataFrame(options,index=[1])
    b =st.button("Click to see your predictions !")
    if b :
        pred =get_predictions(model_name,inputs,X_train,y_train)
        if pred :
         
            st.success('Congratulations !! You have suceedeed!', icon="✅")
           # st.write(f'The prediction of the ML model is ',pred)
        else :
            st.error('tough Luck !! You are out, Next !', icon="🚨")
            #st.write(f'The prediction of the ML model is ',pred)
           # st.write( ' "Cogito ergo sum" Descartes')



if page == pages[3]:
    st.title('My project is not very good')
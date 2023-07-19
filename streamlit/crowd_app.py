import streamlit as st
import pandas as pd
import functions

# Title
main_title=st.markdown("Predicting the succes of crowdfunding campaigns")
main_title

# navigation
st.sidebar.title("Menu")
pages=["Presentation","Visualization","Preprocessing","Modeling",'Conclusions']
page=st.sidebar.radio("Choose a page",options=pages)

# fill pages
if page==pages[0]:
    st.title(pages[0])
    
    # link to raw data
    st.markdown("You may get the raw data here [link](https://www.kaggle.com/yashkantharia/kickstarter-campaigns-dataset-20)")
    
    
    # get data
    filename="Kaggle_deduplicated.csv"
    
    # Create a text element and let the reader know the data is loading.
    data_load_state=st.text('Loading data...')
    df=pd.read_csv(filename) # ,index_col='id'
    df.drop(columns=['Unnamed: 0','id'],inplace=True)
    
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    
    
    # give interactive overview
    st.write('Interactive Table')
    
    # plot dataframe
    st.dataframe(df.style.highlight_min(axis=0))
    
    
    # give static overview
    st.write('static table')
    st.table(df)

if page==pages[1]:
    st.title(pages[1])
    # visual exploration
    # st.write('Bar chart')
    # st.bar_chart(df['status'])
    # goal=st.slider('goal_usd')  # 👈 this is a widget
    # st.write('for this goal, the number of backers ')
    # st.table(df.loc[[df.goal_usd]==goal].backers_count.describe())
    
    # countplot
    countplot=countplot(main_category,status)
    st.pyplot(countplot.fig)
    
    # Interactive plot
    y=st.selectbox('Selection of the data',options=df.columns)
    st.line_chart(df[y])

# Checkboxes

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:16:27 2023

@author: Franz
"""

# --------------------------------------------------------------

### Getting started

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # -> plt.show to plot the respective graph
import seaborn as sns
sns.set() # to change the theme

# show all columns in output
pd.options.display.max_columns=None

# read data from downloaded file
filename=r"C:\Users\Franz.000\Documents\GitHub\MAY23_BDA_INT_Crowdfunding\data\kaggle\Kaggle_Dataset.csv"
raw=pd.read_csv(filename)
raw.head(3)
df=raw

# raw["launched_year"]=pd.to_datetime(raw["launched_at"])
# raw["year"]=raw["launched_year"].dt.year
# raw["year"].value_counts()

# ------------------------------------------------------------------------------


### First inspection

## general overview
# df information
df.info() # 378661 projects

# detect (and delete duplicates)
df.duplicated().sum() # 0
df.id.duplicated().sum() # 22,170
df.drop_duplicates(keep='first',inplace=True,subset='id')
df.id.duplicated().sum() #
len(df) # 192,888 remain

# What categories does the success indicator "status" have?
df.status.value_counts(normalize=True)
sns.countplot(data=df,x="status")
plt.title("Frequencies of status categories");

# reduce to successful and failed projects, (finished projects) only
df=df[(df.status=="successful")|(df.status=="failed")]
len(df) # 180,675 are relevant and retained (after dropping duplicates)
df.status.value_counts(normalize=True) 


## date/time related information
# convert date variables into datetime format, and
# derive year, month, week, and day (of the week) 
from datetime import timedelta
for p in ["year","month","week","wday"]:
    for v,var in zip(["launched","dead"],["launched_at","deadline"]):
        name=v+"_"+p
        # convert into datetime
        df[v]=pd.to_datetime(df[var])
        # derive each of the periods 
        if p=="year":   df[name]=df[v].dt.year
        if p=="month":  df[name]=df[v].dt.month
        if p=="week":   df[name]=df[v].dt.week
        if p=="wday":   df[name]=(df[v].dt.dayofweek)+1 # +1 to make the week start (more readable) with day 1 (instead of 0)
        # print(df[name].value_counts(normalize=True)) # retain day of week at launch

## date-related distributions   
# year        
# original webrobots data frame contained outliers before 2009 (launch of kickstarter platform)
df.launched_year.value_counts() # but: outliers before 2009 already dropped with reduction to finished projects
print(pd.crosstab(df.launched_year,df.status,normalize="index"))
sns.displot(x=df.launched_year,hue=df.status)
plt.xlabel("year a campaign was launched")
plt.title("Distribution of campaign launches over years by status");

# month of launch     
sns.displot(x=df.launched_month,hue=df.status)
plt.xlabel("month a campaign was launched")
plt.title("Distribution of campaign launches over months by status");

# weekday
# first assign string instead of numeric values 
days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
df["launched_day"]=df["launched_wday"].replace(range(1,8),days)
# order according to order of a week
df["launched_day"]=pd.Categorical(df["launched_day"],categories=days,ordered=True)            
sns.countplot(x=df.launched_day,hue=df.status)
plt.xlabel("weekday a campaign was launched")
plt.title("Distribution of campaign launches over weekdays by status");

# duration 
df.duration.value_counts()
sns.displot(df.duration,kde=True)
plt.ylim(0,10000);
plt.xlabel("duration in days")
plt.title("Distribution of campaign durations");


## Main and sub categories
df.describe(include="object")

# reverse column names of category variables
new={"main_category":"sub_category",
"sub_category":"main_category"}
df.rename(columns=new,inplace=True)

# main category
pd.DataFrame(df.main_category.value_counts(normalize=True)[:10])
df.main_category.value_counts()

# sub category
pd.DataFrame(df.sub_category.value_counts(normalize=True))
df.sub_category.value_counts()

# What are the contents (sub categories) of each main category?
for cat in df.main_category.value_counts().index.tolist():
    print("\n",cat,"has following sub categories:\n",df.loc[df["main_category"]==cat].sub_category.value_counts(normalize=True).to_frame())
    
# (sub) category
df.sub_category.value_counts()[:10]
df.sub_category.nunique() # 159 modalities are way too many


## backers - plot calculation takes way too long
# sns.displot(x=df.backers_count,hue=df.status)
# plt.xlabel("number of backers a campaign attracted")
# plt.title("Distribution of backers by status");


## goal - plot calculation takes way too long
# sns.displot(x=df.goal_usd,hue=df.status)
# sns.displot(x=df.goal_usd)
# plt.xlabel("goal height of a campaign")
# plt.title("Distribution of goal height by status");
# plt.title("Distribution of goal height");

df[["backers_count","goal_usd"]].describe()
df.backers_count.value_counts().sort_index(ascending=False)[:20]


## creator
# count projects per creator 
df_creator=pd.DataFrame(df['creator_id'].value_counts())
df_creator['id']=df_creator.index
df_creator['creator_project_counts']=df_creator['creator_id']
df_creator.drop("creator_id",axis=1,inplace=True)
df_creator.rename(columns={"id":"creator_id"},inplace=True)
df_creator

# Building groups
df_creator.loc[df_creator['creator_project_counts']==1,'creator_projects']='1'
df_creator.loc[(df_creator['creator_project_counts']>=2) & (df_creator['creator_project_counts']<=5),'creator_projects']='2-5'
df_creator.loc[(df_creator['creator_project_counts']>=6) & (df_creator['creator_project_counts']<=15),'creator_projects']='6-15'
df_creator.loc[(df_creator['creator_project_counts']>=16) & (df_creator['creator_project_counts']<=30),'creator_projects']='16-30'
df_creator.loc[(df_creator['creator_project_counts']>=31),'creator_projects']='31+'
pd.crosstab(df_creator['creator_project_counts'],df_creator['creator_projects'],dropna=False)

# merge to main dataframe
df=df.merge(right=df_creator,on="creator_id",how="inner")
df['creator_projects'].value_counts()

# Word cloud of blurb
df.blurb.head()

# create wordcloud and map it on the shape of the kickstarter logo
from PIL import Image
import numpy as np
import nltk

from nltk.tokenize import TweetTokenizer
tokenizer=TweetTokenizer()
success=df.loc[df["status"]=="successful"].name.tolist()
fail=df.loc[df["status"]=="failed"].name.tolist()
tokens_success=tokenizer.tokenize(str(success))
len(tokens_success)
tokens_fail=tokenizer.tokenize(str(fail))
len(tokens_fail)

# Display the total number of words as well as the number of different words found in these speeches.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
vectorizer.fit_transform(tokens)
print(vectorizer.fit_transform(tokens).toarray().shape)
len(tokens)

# define unwanted 'words'
from nltk.corpus import stopwords
import re
stop_words=set(stopwords.words('english'))
print(stop_words)
df.country.value_counts()
stop_words.update(["?","!",".",",",":",";","-","--", "...",'"',"'","they've","they're","they'll","i've","i'm", "i'll","could",r"[0-9](\.|,)[0-9]",r"[0-9]*",r"([A-Z]+)[$€¥£]",r"\bzł\b",r"\bCHF\b",r"\bkr\b",r"\bUS([A-Z])\b",r"\+"])
print(stop_words)

# remove stop words
def stop_words_filtering(wordlist):
    for word in wordlist:
        if word in stop_words:
            # print("taking out '",word,"'")
            wordlist.remove(word)  # still some occurrences left, but at least filters some
    # new=""
    # for word in stop_words:
    #     if word in wordlist:
    #         print("taking out '",word,"'")
    #         r=re.compile(f"{word}.?")
    #         new=r.sub("",str(wordlist))
    # wordlist=new
    return wordlist

from time import time
start=time()
tokens_s=stop_words_filtering(tokens_success)
print("processing time (success):",time()-start,"seconds")
start=time()
tokens_f=stop_words_filtering(tokens_fail)
print("processing time (success):",time()-start,"seconds")

[stop_words_filtering(w) for w in df["blurb"].tolist()]

df["blurb_clean"]=df["blurb"].apply(lambda x:' '.join([entry for entry in x.split() if entry not in (stop_words)]))
df.loc[df['status']=='successful'].blurb_clean.unique
df.loc[df['status']=='failed'].blurb_clean.unique


blurbs_success=str(df.loc[df['status']=='successful'].blurb_clean.unique())
blurbs_fail=str(df.loc[df['status']=='failed'].blurb_clean.unique())
for blurb,suffix in zip([blurbs_success,blurbs_fail],["success","fail"]):
    from nltk.tokenize import TweetTokenizer
    tokenizer=TweetTokenizer()
    name="tokens_"+suffix
    name=tokenizer.tokenize(blurb)
    len(name)
# print(blurb_tokens)

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer=CountVectorizer()
    vectorizer.fit_transform(name)
    print(vectorizer.fit_transform(name).toarray().shape)
    len(name)
    print(name)

# Import the WordCloud class from the library wordcloud
from wordcloud import WordCloud

# build function for creation and mapping of the word cloud
def plot_word_cloud(text,mask,background_color="black"):
    """
    This function creates a word cloud and maps on an image.
    """
    
    # Define a mask
    mask_coloring=np.array(Image.open(str(mask)))

    # Define the layer of the word cloud
    wc=WordCloud(background_color=background_color, max_words=200,stopwords=stop_words,mask=mask_coloring, max_font_size=50,random_state=0)
    
    # Generate and display the word cloud
    plt.figure(figsize=(20,10))
    wc.generate(text)
    plt.imshow(wc)
    plt.axis("off")
    plt.show()

plot_word_cloud(df.blurb,"images/kick_logo.png");




## (pre)select features (columns) to retain for further processing
# define selection
list_final=['id','currency','backers_count','country','status', 'usd_pledged','main_category','creator_projects','goal_usd', 'city','launched_year','launched_day','duration']


# reduce dataset to selection
# df_final=df.id
# df_final=pd.concat([df_final,df[list_final]],axis=1)
# df_final=df_final.set_index("id")
# df_final.info()

df_final=df[list_final]
df_final.info()


#---------------------------------------------------------------

### SAVE FILE (directly in GitHub project)
filename=r'data\kaggle\Kaggle_deduplicated.csv'
f=open(filename,'w',encoding='utf-8')
f.write(df_final.to_csv())
f.close()

#---------------------------------------------------------------

from pathlib import Path # check file existence
path=Path(filename)
if path.is_file(): print("finished explore")
else: print("ERROR: file save failed")
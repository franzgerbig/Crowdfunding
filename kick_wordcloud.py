# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:42:23 2023

@author: Franz
"""

### Getting started

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# read data from downloaded file
filename=r"[placeholder]GitHub\MAY23_BDA_INT_Crowdfunding\data\kaggle\Kaggle_Dataset.csv"
raw=pd.read_csv(filename)
raw.head(3)
df=raw
df.drop_duplicates(keep='first',inplace=True,subset='id')

# Word cloud of project names
df.name.head()

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

# define unwanted 'words'
from nltk.corpus import stopwords
import re
stop_words=set(stopwords.words('english'))
print(stop_words)
df.country.value_counts()
stop_words.update(["?","!",".",",",":",";","-","--", "...",'"',"'",r".+'re",r".hey'll",r".+'ve",r"i'm", r".'ll","could",r"[0-9](\.|,)[0-9]",r"[0-9]*",r"([A-Z]+)[$€¥£]",r"\bzł\b",r"\bCHF\b",r"\bkr\b",r"\bUS([A-Z])\b",r"\+",r"project",".?he","[","]",r".+'$"])
print(stop_words)

# remove stop words
tokens_f=[m.lower() for m in tokens_fail if m.lower() not in stop_words]
tokens_s=[m.lower() for m in tokens_success if m.lower() not in stop_words]
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

# now apply stop_words_filtering as often as len(tokens) decreases
tokens_s=stop_words_filtering(tokens_s)
len(tokens_s)
tokens_f=stop_words_filtering(tokens_f)
len(tokens_f)
print(tokens_s[:5])

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
    wc=WordCloud(background_color=background_color, max_words=200,stopwords=stop_words,mask=mask_coloring, max_font_size=100,random_state=0)
    
    # Generate and display the word cloud
    plt.figure(figsize=(10,10))
    wc.generate(text)
    plt.imshow(wc)
    plt.axis("off")
    
    # save wordcloud to disk
    # plt.savefig(f'images/wc_{text}.png',bbox_inches='tight')
    # plt.show();

plot_word_cloud(str(tokens_s),"images/kick_vector_logo.png")
plt.savefig('images/wc_s.png',bbox_inches='tight')
plt.show();
plot_word_cloud(str(tokens_f),"images/kick_vector_logo.png")
plt.savefig('images/wc_f.png',bbox_inches='tight')
plt.show();

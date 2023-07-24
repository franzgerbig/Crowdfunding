# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:46:57 2023

@author: Franz
"""

import streamlit as st
import pandas as pd
import numpy as np


imp="Kaggle_Dataset.csv"
df=pd.read_csv(imp)

# minimal data cleaning
df=df[(df['status']=='successful')|(df['status']=='failed')]
df.drop_duplicates(keep='first',inplace=True,subset='id')
df.drop("Unnamed: 0",axis=1,inplace=True)

# summarize (interesting) numeric variables
sum_country=df.groupby("country").sum()
sum_country.drop(['id','creator_id','blurb_length'],axis=1,inplace=True)

# success (and fail) rate rate per country
success_rate_country=pd.DataFrame(df.groupby("country").status.value_counts(normalize=True)).unstack()
success_rate_country.fillna(0,inplace=True)

# total number of projects per country
projects_country=pd.DataFrame(pd.crosstab(df.country, df.status,margins=True))
projects_country.rename(columns={"All":"projects_count"},inplace=True)
projects_country.drop(["failed","successful"],axis=1,inplace=True)
projects_country=projects_country.loc[projects_country.index!="All"]

# merge
countries=pd.concat([projects_country,success_rate_country,sum_country],axis=1)

# add coordinates (from https://developers.google.com/public-data/docs/canonical/countries_csv?hl=en)
lat_long=pd.read_csv("lat+long.csv",sep=";")
countries=countries.merge(right=lat_long,on="country",how="left")
countries.reset_index(inplace=True)
new={countries.columns[3]:"fail%",countries.columns[4]:"success%"}
countries.rename(columns=new,inplace=True)
countries.drop('index',axis=1,inplace=True)

# save file
filename="countries.csv"
f=open(filename,'w',encoding='utf-8')
f.write(countries.to_csv())
f.close()


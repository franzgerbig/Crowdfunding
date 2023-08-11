
# Predicting the success of a crowdfunding campaign

## Introduction

This repository contains the code for our project **"Kickstarter Success Factors"**, developed during our [Data Analyst training](https://datascientest.com/en/data-analyst-course) at [DataScientest](https://datascientest.com/en/).

### Objective

The goal of this project is to identify
- common characteristics of crowdfunding campaigns, and \n 
- which of those have a positive, and \n 
- which others a negative relation with a campaign's success.

### Team members

This project was developed by the following team:

- Franz Gerbig ([GitHub](https://github.com/franzgerbig) / [LinkedIn](https://linkedin.com/in/franzgerbig))
- Hendrik Bosse ([GitHub](https://github.com/hebosse))

## Try it

### Data Source and preprocessing
The raw data is available directly [here](./data) (or on the data science platform [kaggle](https://www.kaggle.com/yashkantharia/kickstarter-campaigns-dataset-20) without any cost).

To preprocess (preprocessing I) and analyze (preprocessing II & modeling) the data, you can run the [notebooks](./notebooks) - be careful with the filepaths. 

You will need to install (some of) the dependencies (in a dedicated environment):

```
pip install -r requirements.txt
```

### Streamlit App

In a more interactive manner you may play around with the [streamlit app](./streamlit).

To run the app, please execute the following code:

```shell
conda create --name crowdfunding-streamlit python=3.10
conda activate crowdfunding-streamlit
pip install -r requirements.txt
streamlit run streamlit_crowdfunding_BDA_May23.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

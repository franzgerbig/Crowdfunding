# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:15:07 2023

@author: Franz
"""

import seaborn as sns
import matplotlib.pyplot as plt

def countplot(x,hue):
    sns.countplot(x=df[f"{x}"],hue=df[f"{hue}"])
    plt.xlabel(f"{x}")
    plt.title(f"Frequencies of {x} by {hue}");
    

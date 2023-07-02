# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 14:06:02 2023

@authors: Hendrik Bosse, Franz Gerbig
"""

# import modules needed for/in several files
import runpy # import runpy to allow calling of other .py-files


# ---------------------------------------------------------------


### Master file


## exploration
runpy.run_path(path_name='01_kaggle_explore.py')

## visualization 1 (before preprocessing)
runpy.run_path(path_name='02_kaggle_visualize_1.py')

## preprocessing
runpy.run_path(path_name='03_kaggle_preprocess.py')

## visualization 2 (after preprocessing)
runpy.run_path(path_name='04_kaggle_visualize_2.py')

## modeling  
runpy.run_path(path_name='05_kaggle_modeling.py')

## optimizing
runpy.run_path(path_name='06_kaggle_optimize_modeling.py')


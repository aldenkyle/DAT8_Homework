# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:19:40 2015

@author: kylealden
"""
## Iris working
import pandas as pd
import matplotlib.pyplot as plt

#import iris dataset
iris_cols = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'species']
iris_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=iris_cols, header=None, sep = ",")

def irisfun(iris_data):
    #calculate area variables for petal and sepal using function for area of elipse
    iris_data['petal_area'] = 3.14 * (iris_data.petal_len/2) * (iris_data.petal_wid/2)
    iris_data['sepal_area'] = 3.14 * (iris_data.sepal_len/2) * (iris_data.sepal_wid/2)
    #ksa_predict for versicolor first (like else but will calculate over it)
    iris_data['ksa_predict'] = "Iris-versicolor"
    #ksa_predict for iris setosa if petal area is less than 2
    iris_data['ksa_predict'][(iris_data['petal_area'] < 2)] = "Iris-setosa"
    #ksa_predict for iris virginica if petal area is less than 5.87 but sepal area greater than 14.82
    iris_data['ksa_predict'][(iris_data['petal_area'] > 5.87) & (iris_data['sepal_area'] < 14.82)] = "Iris-virginica"
    #ksa_predict for iris virginica if petal area is greater than 7
    iris_data['ksa_predict'][(iris_data['petal_area'] > 7)] = "Iris-virginica"

#run_function
irisfun(iris_df)   
 
#check to see how many rows were correct
iris_df[iris_data.species == iris_data.ksa_predict].shape

#calculate percentage correct of 150 rows in iris_data dataframe
149/150.0

## irisfun predicted 99.3 percent of the iris species 


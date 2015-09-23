# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:14:19 2015

@author: kylealden
"""

# read the data and set the datetime as the index
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv'
bikes = pd.read_csv(url, index_col='datetime', parse_dates=True)

bikes.head()

# "count" is a method, so it's best to name that column something else
bikes.rename(columns={'count':'total'}, inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# Pandas scatter plot
bikes.plot(kind='scatter', x='temp', y='total', alpha=0.2)

sns.lmplot(x='temp', y='total', data=bikes, aspect=1.5, scatter_kws={'alpha':0.2})


# create X and y
feature_cols = ['temp']
X = bikes[feature_cols]
y = bikes.total


# import, instantiate, fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
linreg.intercept_
linreg.coef_

### _ represents an attribute that is only known after fitting a model









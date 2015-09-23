# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:42:26 2015

@author: kylealden
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 9

# 1. Read yelp.csv into a DataFrame.
yelp = pd.read_csv('yelp.csv', header=0)

yelp.head()
yelp.columns
yelp.dtypes


# 2. Explore the relationship between each of the vote types (cool/useful/funny) and the number of stars.

yelp.plot(kind='scatter', x='stars', y='cool')
### slight positive relationship between stars and cool
 
yelp.plot(kind='scatter', x='stars', y='useful')
### slight positive relationship between stars and cool

yelp.plot(kind='scatter', x='stars', y='funny')
### less clear relationship between stars and cool


# 3. Define cool/useful/funny as the features, and stars as the response.
 feature_cols = ['cool','useful','funny']
 X = yelp[feature_cols]
 X.head()
 ###chek type and shape (Data Frame -- 10000, 3 )
type(X)
X.shape 
 
y = yelp['stars']
y.head()
###chek type and shape
type(y)
y.shape 


# 4. Fit a linear regression model and interpret the coefficients. 
#Do the coefficients make intuitive sense to you? 
#Explore the Yelp website to see if you detect similar trends.
# import, instantiate, fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)

linreg.intercept_
### intercept == 3.84

linreg.coef_

### coefs = array([ 0.27435947, -0.14745239, -0.13567449])

### I wasnt surprised by the positive relationship between cool and stars, but was 
### by the negative relationship between useful and stars. A seaborn plot with line would
### have helped me see that more cleanly. 

# 5. Evaluate the model by splitting it into training and testing sets and computing the RMSE. Does the RMSE make intuitive sense to you?

# define a function that accepts a list of features and returns testing RMSE

from sklearn import metrics
import numpy as np
def train_test_rmse(feature_cols):
    X = yelp[feature_cols]
    y = yelp.stars
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


train_test_rmse(['cool', 'useful', 'funny'])

### with all three we get 1.18 as RMSE





# 6. Try removing some of the features and see if the RMSE improves.
train_test_rmse(['cool', 'useful'])

### RMSE for Cool + Useful was 1.20
train_test_rmse(['cool', 'funny'])
### RMSE for Cool + Useful was 1.19

train_test_rmse(['useful', 'funny'])
### RMSE for Cool + Useful was 1.21

train_test_rmse(['funny'])

##I tried each feature by itself and none performed better than the group of three

yelp['txt_length'] = yelp.text.str.len()

yelp.text.str.len()

yelp.head()


yelp.plot(kind='scatter', x='stars', y='txt_length')


train_test_rmse(['cool', 'useful', 'funny', 'txt_length'])

# 7. Bonus: Think of some new features you could create from the 
#existing data that might be predictive of the response. Figure out 
#how to create those features in Pandas, add them to your model, and see 
#if the RMSE improves.

### Create new feature with length of the text

yelp['txt_length'] = yelp.text.str.len()

yelp.text.str.len()

yelp.head()

### check it out
yelp.plot(kind='scatter', x='stars', y='txt_length')
### no clear relationship

train_test_rmse(['cool', 'useful', 'funny', 'txt_length'])

### RMSE of 1.18 again, a very small improvement




# 8. Bonus: Compare your best RMSE on the testing set with the RMSE 
#for the "null model", which is the model that ignores all features 
#and simply predicts the mean response value in the testing set.

### split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

### create a NumPy array with the same shape as y_test
y_null = np.zeros_like(y_test, dtype=float)

### fill the array with the mean value of y_test
y_null.fill(y_test.mean())
y_null


### compute null RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_null))

### NULL RMSE is 1.21, making our model slightly better

# 9. Bonus: Instead of treating this as a regression problem, treat 
#it as a classification problem and see what testing accuracy you
# can achieve with KNN.

# import the class
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=32)

# train the model on the entire dataset
knn.fit(X, y)

# predict the response values for the observations in X ("test the model")
knn.predict(X)


y_pred_class = knn.predict(X)

# compute classification accuracy
from sklearn import metrics
metrics.accuracy_score(y, y_pred_class)

# Now with train test split
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# train the model on the training set
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train, y_train)

y_pred_class = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

### Using train test split the best accuracy I came up with was about 35%

# 10. Bonus: Figure out how to use linear regression for classification, 
#and compare its classification accuracy with KNN's accuracy.


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:01:03 2015

@author: kylealden
"""
# read the iris data into a DataFrame
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)

iris.head()

# allow plots to appear in the notebook
%matplotlib inline
import matplotlib.pyplot as plt

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

# create a custom colormap
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# map each iris species to a number
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# create a scatter plot of PETAL LENGTH versus PETAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap=cmap_bold)

# create a scatter plot of SEPAL LENGTH versus SEPAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='sepal_length', y='sepal_width', c='species_num', colormap=cmap_bold)

'''
Requirements for working with data in scikit-learn
Features and response should be separate objects
Features and response should be entirely numeric
Features and response should be NumPy arrays (or easily converted to NumPy arrays)
Features and response should have specific shapes (outlined below)'''

# store feature matrix in "X"
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[feature_cols]

# alternative ways to create "X"
### everything but
X = iris.drop(['species', 'species_num'], axis=1)
### .loc refer to rows and columns by name
X = iris.loc[:, 'sepal_length':'petal_width']
### .iloc refer to rows and columns by index number
X = iris.iloc[:, 0:4]

# store response vector in "y"
y = iris.species_num

# check X's type
print type(X)
print type(X.values)
# check y's type
print type(y)
print type(y.values)

# check X's shape (n = number of observations, p = number of features)
print X.shape

# check y's shape (single dimension with length n)
print y.shape


'''
scikit-learn's 4-step modeling pattern
Step 1: Import the class you plan to use'''

from sklearn.neighbors import KNeighborsClassifier

'''
Step 2: "Instantiate" the "estimator"
"Estimator" is scikit-learn's term for "model"
"Instantiate" means "make an instance of"'''

# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=1)
type(knn)


'''
- Created an object that "knows" how to do K-nearest neighbors classification, and is just waiting for data
- Name of the object does not matter
- Can specify tuning parameters (aka "hyperparameters") during this step
- All parameters not specified are set to their defaults'''


print knn

'''
Step 3: Fit the model with data (aka "model training")
Model is "learning" the relationship between X and y in our "training data"
Process through which learning occurs varies by model
Occurs in-place '''

knn.fit(X, y)

### Once a model has been fit with data, it's called a "fitted model"

'''Step 4: Predict the response for a new observation
New observations are called "out-of-sample" data
Uses the information it learned during the model training process'''

knn.predict([3, 5, 4, 2])


'''Tuning a KNN model'''


# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)
​
# fit the model with data
knn.fit(X, y)
​
# predict the response for new observations
knn.predict(X_new)

# calculate predicted probabilities of class membership
knn.predict_proba(X_new)




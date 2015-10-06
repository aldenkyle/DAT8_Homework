# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 9

# 1. Read yelp.csv into a DataFrame.

#Read yelp.csv into a DataFrame.

yelp = pd.read_csv('yelp.csv', header=0)

yelp.head()
yelp.columns
yelp.dtypes

#2. Create a new DataFrame that only contains the 5-star and 1-star reviews.

yelp5_1 = yelp[(yelp.stars == 1) | (yelp.stars == 5)]

#3. Split the new DataFrame into training and testing sets, using the review text
#as the only feature and the star rating as the response.
# define X and y
X = yelp5_1.text
y = yelp5_1.stars

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)

#4. Use CountVectorizer to create document-term matrices from X_train and X_test.
#Hint: If you run into a decoding error, instantiate the vectorizer with 
#the argument decode_error='ignore'.

from sklearn.feature_extraction.text import CountVectorizer
# instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then create document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm

#5. Use Naive Bayes to predict the star rating for reviews in the testing set, 
#and calculate the accuracy.

# train a Naive Bayes model using X_train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

### Accuracy was 91.8%

#6. Calculate the AUC.
#Hint 1: Make sure to pass the predicted probabilities to roc_auc_score,
# not the predicted classes.
#Hint 2: roc_auc_score will get confused if y_test contains fives 
#and ones, so you will need to create a new object that contains 
#ones and zeros instead.

# predict (poorly calibrated) probabilities
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob

##create binary test
y_test_binary = (y_test -1)/4
y_test_binary

metrics.roc_auc_score(y_test_binary, y_pred_prob)


### AUC was .94

#7. Plot the ROC curve.

# plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test_binary, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


#8. Print the confusion matrix, and calculate the sensitivity and specificity. 
#Comment on the results.


metrics.confusion_matrix(y_test, y_pred_class)


'''array([[126,  58],
       [ 25, 813]] '''
 
sensitivity = 813/838
specificity = 126 / (126+58)  

specificity
### KSA Comment: Sensitivity = .97 , specificity = .68 , our model did well at prediciting 5s 
### but not as well at predicting 1s (though this is confusing, )


#9. Browse through the review text for some of the false positives and false negatives. 
#Based on your knowledge of how Naive Bayes works, do you have any theories about why
#the model is incorrectly classifying these reviews?

# print message text for the false negatives
X_test[y_test > y_pred_class]

# print message test for the false positives
X_test[y_test < y_pred_class]

### seems to have a lot of ! pts, though they're surprising, really not sure


#10. Let's pretend that you want to balance sensitivity and specificity. 
#You can achieve this by changing the threshold for predicting a 5-star review.
# What threshold approximately balances sensitivity and specificity?

# histogram of predicted probabilities grouped by actual response value
df = pd.DataFrame({'probability':y_pred_prob, 'actual':y_test})
df.hist(column='probability', by='actual', sharex=True, sharey=True)


fpr, tpr, thresholds = metrics.roc_curve(y_test_binary, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


### KSA Comment:  a threshold of .9 or higher might have helped, I think I understand
### what this means, but I have no clue how to do it. 

#11. Let's see how well Naive Bayes performs when all reviews are included, 
#rather than just 1-star and 5-star reviews:
#Define X and y using the original DataFrame from step 
#1. (y should contain 5 different classes.)
#Split the data into training and testing sets.
#Calculate the testing accuracy of a Naive Bayes model.
#Compare the testing accuracy with the null accuracy.
#Print the confusion matrix.
#Comment on the results.



X = yelp.text
y = yelp.stars

#Split the data into training and testing sets.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)


from sklearn.feature_extraction.text import CountVectorizer
# instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then create document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# train a Naive Bayes model using X_train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

### KSA Comment: Testing accuracy is .47', this is better than the null (which I think is
### .2 because we would randomly guess 1 in 5 correcly), but not amazing. 



metrics.confusion_matrix(y_test, y_pred_class)

'''array([[ 55,  14,  24,  65,  27],
       [ 28,  16,  41, 122,  27],
       [  5,   7,  35, 281,  37],
       [  7,   0,  16, 629, 232],
       [  6,   4,   6, 373, 443]]'''
       
### KSA Comment -- it looks like though we weren't perfect, we often classified 
### star rankings one number off



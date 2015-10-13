# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:21:58 2015

@author: kylealden
"""

import pandas as pd
import numpy as np
train = pd.read_csv('train.csv', index_col =0)

train.head()

train.shape

train.OpenStatus.value_counts()

##Balanced dataset, strange because only 6%
## end up closed, artificially balanced

train.OwnerUserId.value_counts()

train[train.OwnerUserId==466534].describe()
train[train.OwnerUserId==466534].head()


train.groupby('OwnerUserId').OpenStatus.mean()

train.groupby('OwnerUserId').OpenStatus.agg(['mean', 'count']).sort('count')

train[train.OwnerUserId==185593].describe()
train[train.OwnerUserId==185593].head()



train.groupby('OpenStatus').ReputationAtPostCreation.describe().unstack()

train.ReputationAtPostCreation.plot(kind='hist')

train[train.ReputationAtPostCreation < 1000].ReputationAtPostCreation.plot(kind='hist')

train[train.ReputationAtPostCreation < 1000].hist(column='ReputationAtPostCreation', by='OpenStatus')

train[train.ReputationAtPostCreation < 1000].boxplot(column='ReputationAtPostCreation', by='OpenStatus')

train.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)

train.groupby('OpenStatus').Answers.describe().unstack()

train['TitleLength']= train.Title.apply(len)

train.head()

train.groupby('OpenStatus').TitleLength.describe().unstack()

### Make a function to do the above
def make_features(filename):
    df = pd.read_csv(filename, index_col = 0) 
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength']= df.Title.apply(len)
    return df
    
## Apply function to training an test sets
train = make_features('train.csv')
test = make_features('test.csv')

##Actually build a model
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength' ]

X = train[feature_cols]
y = train.OpenStatus

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

logreg.coef_

### do these coefs make sense
''' array([[  2.33237069e-05,   1.75473166e-03,   1.05731957e-02]]''' 
### 
### all coefs are pos, reputation is least important, 

y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]


### what are all the metrics we've used so far?
# accuracy
# confustion matrix
# ROC curve / AUC
# specificity, sensitivity

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
metrics.confusion_matrix(y_test, y_pred_class)
metrics.roc_auc_score(y_test, y_pred_prob)
metrics.log_loss(y_test, y_pred_prob)


logreg.fit(X,y)

X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:,1]

oos_pred_prob.shape

test.head(1)

### get submission out
sub = pd.DataFrame({'id': test.index, 'OpenStatus': oos_pred_prob}).set_index('id')
sub.to_csv('sub1.csv')


### working on adding new features

def make_features(filename):
    df = pd.read_csv(filename, index_col = 0, parse_dates = ['OwnerCreationDate', 'PostCreationDate']) 
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength']= df.Title.apply(len)
    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1) 
    df['OwnerAge'] = (df.PostCreationDate - df.OwnerCreationDate).dt.days
    df['OwnerAge'] = np.where(train.OwnerAge < 0, 0, train.OwnerAge)
    return df


## find number of tags
train['NumTags'] = train.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1) 


train.dtypes
train.Tag1.head()


train.groupby('OpenStatus').NumTags.describe().unstack()
train.groupby('NumTags').OpenStatus.mean()

train.groupby('OpenStatus').NumTags.describe().unstack()


train.groupby('OpenStatus').OwnerUserID

train['OwnerCreationDate'] = pd.to_datetime(train.OwnerCreationDate)
train['PostCreationDate'] =pd.to_datetime(train.PostCreationDate)

train.dtypes

train['OwnerAge'] = (train.PostCreationDate - train.OwnerCreationDate).dt.days

train.groupby('OpenStatus').OwnerAge.describe().unstack()
train.boxplot(column = 'OwnerAge', by = 'OpenStatus')
import numpy as np
np.where(train.OwnerAge < 0, 0, train.OwnerAge)


def make_features(filename):
    df = pd.read_csv(filename, index_col = 0, parse_dates = ['OwnerCreationDate', 'PostCreationDate']) 
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength']= df.Title.apply(len)
    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1) 
    df['OwnerAge'] = (df.PostCreationDate - df.OwnerCreationDate).dt.days
    df['OwnerAge'] = np.where(df.OwnerAge < 0, 0, df.OwnerAge)
    return df



## Apply function to training an test sets
train = make_features('train.csv')
test = make_features('test.csv')


##Actually build a model
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'NumTags', 'OwnerAge' ]

X = train[feature_cols]
logreg.fit(X,y)

## predict class probas for the actual test
X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:,1]

sub = pd.DataFrame({'id': test.index, 'OpenStatus': oos_pred_prob}).set_index('id')
sub.to_csv('sub2.csv')


# use CountVectorizer with the default settings
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
dtm = vect.fit_transform(train.Title)

# define X and y
X = dtm
y = train.OpenStatus

# slightly improper cross-validation of a Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
cross_val_score(nb, X, y, scoring='log_loss', cv=10).mean()    # 0.657

# try tuning CountVectorizer and repeat Naive Bayes
vect = CountVectorizer(stop_words='english')
dtm = vect.fit_transform(train.Title)
X = dtm
cross_val_score(nb, X, y, scoring='log_loss', cv=10).mean()    # 0.635

# build document-term matrix for the actual testing data and make predictions
nb.fit(X, y)
oos_dtm = vect.transform(test.Title)
oos_pred_prob = nb.predict_proba(oos_dtm)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub3.csv')  # 0.544


# use CountVectorizer with the default settings
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
dtm = vect.fit_transform(train.BodyMarkdown)

# define X and y
X = dtm
y = train.OpenStatus

# slightly improper cross-validation of a Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
cross_val_score(nb, X, y, scoring='log_loss', cv=10).mean()    # 0.657

# try tuning CountVectorizer and repeat Naive Bayes
vect = CountVectorizer(stop_words='english')
dtm = vect.fit_transform(train.BodyMarkdown)
X = dtm
cross_val_score(nb, X, y, scoring='log_loss', cv=10).mean()    # 0.635

# build document-term matrix for the actual testing data and make predictions
nb.fit(X, y)
oos_dtm = vect.transform(test.BodyMarkdown)
oos_pred_prob = nb.predict_proba(oos_dtm)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub4.csv')  # 0.544



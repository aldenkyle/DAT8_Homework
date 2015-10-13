# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:38:57 2015

@author: kylealden
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 9

counties = pd.read_csv('la.data.64.County.txt', header=0, sep = '\t')

counties.head()
counties.columns
## u'series_id', u'year', u'period', u'value', u'footnote_codes',
counties.shape 
## (4272544, 6)
counties.dtypes


### create a df that just includes data after 2006 and where the last char is 3
### last char of 3 means that its the unemployment rate (others are emlpoyed or pop that could be employed)

### create a series with the last dig of series_id

counties['seriesLastDig'] = counties.series_id.str.slice(start=19)
counties.head(350)
counties['seriesLastDig'] = str(counties['seriesLastDig'])


##create a new df that only includes data after 2006 and where the last dig was 3
countiesRecent = counties[(counties.year > 2006)]
countiesRecent.head(30)
countiesRecent.shape 
##  (1428924, 6)
countiesRecent3 = countiesRecent[(countiesRecent.seriesLastDig == '3 ')]

countiesRecent3 = countiesRecent[(countiesRecent.series_id.str.slice(start=19, stop=20) == '3')]
countiesRecent3.head(300)
countiesRecent3.shape 
###(357231, 6)

### create variable for year and month
countiesRecent3.drop('YearMonth3', inplace=True)
countiesRecent3['YearMonth3'] = str(countiesRecent3['year']) + str(countiesRecent3['period'])
countiesRecent3['Month'] = countiesRecent3.period.str.replace('M','')
countiesRecent3['Month2'] = int(countiesRecent3.Month)

countiesRecent3['YearMonth3'] = countiesRecent3.year*100  + int(countiesRecent3.Month)

countiesRecent3['YearMonth4'] = str(countiesRecent3.YearMonth3) & countiesRecent3.Month

countiesRecent3.dtypes

countiesRecent3[['year']] = countiesRecent[['year']].astype(str)
countiesRecent3[['period']] = countiesRecent[['period']].astype(str)

countiesRecent3.head()


##ufo[ufo.State == 'VA'].City.describe()
countiesRecent = counties[counties.year > 2006]

countiesRecent.head() 

countiesRecent.columns
###: pivoted = df.pivot('date', 'variable')

### write new recent file to csv
countiesRecent.to_csv('CountiesRecent.csv')
countiesRecent.drop(YearMonth4)

df.pivot(index='date', columns='variable', values='value')
cty_pivot =  countiesRecent.pivot(index='series_id', columns='YearMonth3', values='value')

cty_pivot.head()

cty_pivot.head()
cty_pivot.columns
cty_pivot.shape 

cty_pivot.to_csv('Cty_pivot.csv')


### Used sheets to find the number of months it until return to pre-recession rate
### found average of 2007
### created 3 month moving average from 2009 on
### used =INDEX(AM$1:DN$1,MATCH(TRUE,INDEX(AM4:DN4<C4),0)) to get the column name of
### the month when the 3 month moving average dipped below 2007 average
### added FIPS and FIPS_code columns by splitting out sections of the data

'''Part 2 Read Data Back into Pandas'''
cnty_data = pd.read_csv('CntyDataClean_v9.csv', header=0)

cnty_data.head()
cnty_data.columns
cnty_data.dtypes

### Drop Counties that have nulls for regression problem
cnty_data_notnull = cnty_data[cnty_data.MonthsReturn_pls1.notnull() & cnty_data.mine.notnull() & cnty_data.Pct_HS_Grad_only_2009_2013.notnull()] 
cnty_data_notnull.isnull().sum()
cnty_data_notnull.shape
cnty_data_notnull.dtypes
     
### Drop Counties that have nulls for classification problem     
cnty_data_notnull_cl = cnty_data[cnty_data.mine.notnull() & cnty_data.Pct_HS_Grad_only_2009_2013.notnull()] 
cnty_data_notnull_cl.isnull().sum()
cnty_data_notnull_cl.shape


'''Part 3 Data Exploration'''

## Plot histogram of months til return within 1%

cnty_data_notnull[cnty_data_notnull.MonthsReturn_pls1 > 0].MonthsReturn_pls1.plot(kind='hist', bins=80)
plt.xlabel('Months until return within 1% of 2007 Ave Unemployment Rate')
plt.ylabel('Number of Counties')
plt.savefig('counties_Return_hist.png')
## box plot of months until return within 1%
cnty_data_notnull[cnty_data_notnull.MonthsReturn_pls1 > 0].MonthsReturn_pls1.plot(kind='box')
## plot fips codes vs months (quick spatial look)


import seaborn as sns
cnty_data_notnull[cnty_data_notnull.MonthsReturn_pls1 > 0].plot(kind='scatter', x='PctBlack', y='MonthsReturn_pls1')

sns.lmplot(x='Pct_less_than_hs_2009_2013', y='MonthsReturn_pls1', data=cnty_data, aspect=1.5, scatter_kws={'alpha':0.2})


cnty_data[(cnty_data.MonthsReturn_pls1 > 0)].plot(kind='scatter', x='LON', y='LAT', c='MonthsReturn_pls1')

cnty_data[(cnty_data.MonthsReturn_pls1 > 0) & (cnty_data.LON<0)].plot(kind='scatter', x='LON', y='LAT', c='MonthsReturn_pls1', colormap='Reds')

## Map of Continental US by Months until return
cnty_data[(cnty_data.MonthsReturn_pls1 > 0) & (cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='MonthsReturn_pls1', colormap='Reds')
plt.savefig('counties_map_continental.png')
##Map of Stimulus
cnty_data[ (cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='StimulusPerCapita', colormap='Reds')
plt.savefig('counties_map_stimulus.png')

##Map of Stimulus
cnty_data[ (cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Pct_college_grad_2009_2013', colormap='Reds')
plt.savefig('counties_map_pctCollegeGrad.png')

##Map of Rural Urba
cnty_data[(cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Urban_influence_code_2013', colormap='Reds')
plt.savefig('counties_map_Rural_Urban.png')

cnty_data[(cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Rural_urban_continuum_code_2013', colormap='Reds')
plt.savefig('counties_map_Rural_Urban.png')


cnty_data[(cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Rural_urban_continuum_code_2013', colormap='Reds')
plt.savefig('counties_map_Rural_Urban.png')

cnty_data[cnty_data.MonthsReturn_pls1 > 0].MonthsReturn_pls1.value_counts().plot(kind='bar')



cnty_data = pd.read_csv('CntyDataClean_v9.csv', header=0

### Explore Binary Variables
## Industry Policy Variables
feature_cols = ['farm', 'serv', 'mine', 'manf', 'fsgov']
sns.pairplot(cnty_data_notnull, x_vars=feature_cols, y_vars='MonthsReturn_pls1', kind='reg')
plt.savefig('cnty_industry_code_pt1.png')

feature_cols = ['house', 'loweduc', 'lowemp']
sns.pairplot(cnty_data_notnull, x_vars=feature_cols, y_vars='MonthsReturn_pls1', kind='reg')
plt.savefig('cnty_code_pt1.png')

feature_cols = ['perpov', 'poploss', 'rec']
sns.pairplot(cnty_data_notnull, x_vars=feature_cols, y_vars='MonthsReturn_pls1', kind='reg')
plt.savefig('cnty_code_pt2.png')

feature_cols = ['retire', 'perchldpov']
sns.pairplot(cnty_data_notnull, x_vars=feature_cols, y_vars='MonthsReturn_pls1', kind='reg')
plt.savefig('cnty_code_pt3.png')

## Seasonal Variables
feature_cols = ['Summer678', 'Spring345', 'Fall91011', 'HolidaySeason']
sns.pairplot(cnty_data_notnull, x_vars=feature_cols, y_vars='MonthsReturn_pls1', kind='reg')
plt.savefig('SeasonCodes.png')

# visualize correlation matrix in Seaborn using a heatmap
sns.heatmap(cnty_data_notnull.corr())
plt.savefig('CorrelMatrix.png')

sns.heatmap(cnty_data_notnull_cl.corr())
plt.savefig('ClassCorrelMatrix.png')

'''Part 4 Modeling '''
##Regression modeling --- Linear Regression

from sklearn.cross_validation import train_test_split

feature_cols = ['serv', 'farm', 'HolidaySeason']
# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(feature_cols):
    X = cnty_data_notnull[feature_cols]
    y = cnty_data_notnull.MonthsReturn_pls1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

## Attempt 1
feature_cols = ['serv', 'farm', 'HolidaySeason']
train_test_rmse(feature_cols)
# RMSE = 17.10

## Attempt 2
feature_cols = ['serv', 'farm', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'July2015', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']
train_test_rmse(feature_cols)
## RMSE = 13.749


# Calcularte Null RMSE
# split X and y into training and testing sets
X = cnty_data_notnull[feature_cols]
y = cnty_data_notnull.MonthsReturn_pls1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# create a NumPy array with the same shape as y_test
y_null = np.zeros_like(y_test, dtype=float)

# fill the array with the mean value of y_test
y_null.fill(y_test.mean())
y_null
# compute null RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_null))
# define a function that accepts a list of features and returns testing RMSE

### NULL RMSE = 18.63






###Classification modeling --- Logistic Regression

feature_cols = ['serv', 'farm', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'July2015', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



def train_test_auc_score(feature_cols):
    X = cnty_data_notnull_cl[feature_cols]
    y = cnty_data_notnull_cl.Returned
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    logreg = LogisticRegression(C=1e9)
    logreg.fit(X_train, y_train)
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    return metrics.roc_auc_score(y_test, y_pred_prob)


def train_test_confu_matrix(feature_cols):
    X = cnty_data_notnull_cl[feature_cols]
    y = cnty_data_notnull_cl.Returned
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    logreg = LogisticRegression(C=1e9)
    logreg.fit(X_train, y_train)
    y_pred_class = logreg.predict(X_test)
    return metrics.confusion_matrix(y_test, y_pred_class



# Attempt 1
feature_cols = ['serv', 'farm', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'July2015', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']
train_test_auc_score(feature_cols)
train_test_confu_matrix(feature_cols)
### AUC = .94
### Confusion Matrix
##[ 43,  44]     
##[ 11, 646]]














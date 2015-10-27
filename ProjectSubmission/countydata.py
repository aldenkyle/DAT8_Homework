# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:38:57 2015

@author: kylealden
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 8

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
cnty_data = pd.read_csv('Cnty_Results_v15.csv', header=0)

cnty_data.head()
cnty_data.columns
cnty_data.dtypes

### Drop Counties that have nulls for regression problem
cnty_data_notnull = cnty_data[(cnty_data.MonthsReturn_pls1 > 0) & cnty_data.mine.notnull() & cnty_data.Pct_HS_Grad_only_2009_2013.notnull()] 
cnty_data_notnull.isnull().sum()
cnty_data_notnull.shape
cnty_data_notnull.dtypes
     
### Drop Counties that have nulls for classification problem     
cnty_data_notnull_cl = cnty_data[cnty_data.mine.notnull() & cnty_data.Pct_HS_Grad_only_2009_2013.notnull()] 
cnty_data_notnull_cl.isnull().sum()
cnty_data_notnull_cl.shape
cnty_data_notnull.dtypes

### Drop Counties that have nulls for 3 class classification problem     
cnty_data_notnull_cl2 = cnty_data[cnty_data.mine.notnull() & cnty_data.Pct_HS_Grad_only_2009_2013.notnull()] 
cnty_data_notnull_cl2.isnull().sum()
cnty_data_notnull_cl2.shape
cnty_data_notnull.dtypes

### Drop Counties that have nulls for regression problem with limited cols
cnty_data_notnull_rg2 = cnty_data[cnty_data.MonthsReturn_pls1.notnull() & cnty_data.mine.notnull() & cnty_data.Pct_HS_Grad_only_2009_2013.notnull() & (cnty_data.ReturnClass2 == 2)] 
cnty_data_notnull_rg2.isnull().sum()
cnty_data_notnull_rg2.shape
cnty_data_notnull_rg2.dtypes
## (2494, 72)
'''Part 3 Data Exploration'''

## Plot histogram of months til return within 1%

cnty_data_notnull[cnty_data_notnull.MonthsReturn_pls1 > 0].MonthsReturn_pls1.plot(kind='hist', bins=80)
plt.xlabel('Months until return within 1% of 2007 Ave Unemployment Rate')
plt.ylabel('Number of Counties')
plt.savefig('counties_Return_hist.png')

### Plot histogram of months
cnty_data_notnull_rg2.Month.plot(kind='hist', bins=12)
plt.xlabel('Months until return within 1% of 2007 Ave Unemployment Rate')
plt.ylabel('Number of Counties')
plt.savefig('month_hist.png')
## box plot of months until return within 1%
cnty_data_notnull[cnty_data_notnull.MonthsReturn_pls1 > 0].MonthsReturn_pls1.plot(kind='box')
## plot fips codes vs months (quick spatial look)


import seaborn as sns
cnty_data_notnull[cnty_data_notnull.MonthsReturn_pls1 > 0].plot(kind='scatter', x='PctBlack', y='MonthsReturn_pls1')

sns.lmplot(x='Pct_less_than_hs_2009_2013', y='MonthsReturn_pls1', data=cnty_data, aspect=1.5, scatter_kws={'alpha':0.2})


cnty_data[(cnty_data_notnull.MonthsReturn_pls1 > 0)].plot(kind='scatter', x='LON', y='LAT', c='MonthsReturn_pls1')

cnty_data[(cnty_data_notnull.MonthsReturn_pls1 > 0) & (cnty_data.LON<0)].plot(kind='scatter', x='LON', y='LAT', c='MonthsReturn_pls1', colormap='Reds')

## Map of Continental US by Months until return
cnty_data_notnull[(cnty_data_notnull.MonthsReturn_pls1 > 0) & (cnty_data_notnull.LON<0) & (cnty_data_notnull.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='MonthsReturn_pls1', colormap='Reds')
plt.savefig('counties_map_continental.png')

## Map of Continental US by Months until return
cnty_data_notnull_cl2[(cnty_data_notnull_cl2.LON<0) & (cnty_data_notnull_cl2.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='ReturnClass2', colormap='YlGnBu')
plt.savefig('counties_mapclass_continental.png')


## Map of Continental US by Months until return
cnty_data_notnull_cl[(cnty_data_notnull_cl.LON<0) & (cnty_data_notnull_cl.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Returned', colormap='Reds')
plt.savefig('counties_returned_map_continental.png')

## Map of Oil/Gas Prod Locations in US by Months until return
cnty_data_notnull_cl[(cnty_data_notnull_cl.LON<0) & (cnty_data_notnull_cl.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='OilRegion', colormap='Reds')
plt.savefig('counties_returned_map_continental.png')

## Map of Coal Prod Locations in US
cnty_data_notnull_cl[(cnty_data_notnull_cl.LON<0) & (cnty_data_notnull_cl.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='coalBinay', colormap='Reds')
plt.savefig('counties_coal_continental.png')



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

## Map of Continental US by Months until return
cnty_data_notnull.plot(kind='scatter', x='Month', y='StimPC_Limit', c='MonthsReturn_pls1', colormap='Reds')
plt.savefig('counties_Month_vs_stimulus.png')

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


feature_cols= ['MonthsReturn_pls1', 'LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']
sns.heatmap(cnty_data_notnull_rg2[feature_cols].corr())
plt.savefig('CorrelMatrix.png')

sns.heatmap(cnty_data_notnull_cl.corr())
plt.savefig('ClassCorrelMatrix.png')


feature_cols_all = ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus']

'''Part 4 Modeling '''
##Regression modeling -- How many months until a county returns to pre-recession unemp?

## --- Linear Regression
from sklearn.linear_model import LinearRegression
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
# RMSE = 20.7

## Attempt 2
feature_cols = ['serv', 'farm', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'July2015', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']
train_test_rmse(feature_cols)
## RMSE = 19.97


#Attempt 3
feature_cols = ['serv', 'farm', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013', 'January','February', 'April', 'May', 'October', 'November']
train_test_rmse(feature_cols)

## RMSE = 18.5

#Attempt 4
feature_cols = ['serv', 'farm', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013', 'January','February', 'April', 'May', 'October', 'November', 'OilRegion', 'coalBinay']
train_test_rmse(feature_cols)
## RMSE = 14.88

#Attempt 5 (ALL!)
feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']
train_test_rmse(feature_cols)
## RMSE = 14.27

#Attempt 5 (Null)
feature_cols= ['Average2007']
train_test_rmse(feature_cols)
## RMSE = 18.31


# Calculate Null RMSE
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

### NULL RMSE = 18.773

------------------------------------------------------------------------
#Polynomial Regression (linear but cast the values)

cols = ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']

featuresPoly = cnty_data_notnull['MonthsReturn_pls1', cols]

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
poly = PolynomialFeatures(degree=3)
featuresPoly2 = poly.fit_transform(featuresPoly)

featuresPoly2.shape()

feature_cols= ['Average2007']
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
model = Pipeline([('poly', PolynomialFeatures(degree=5)), ('linear', LinearRegression(fit_intercept=False))])
 # fit to an order-3 polynomial data
X = cnty_data_notnull[feature_cols]
y = cnty_data_notnull.MonthsReturn_pls1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


model.named_steps['linear'].coef_

feature_cols= ['Average2007']
## RMSE = 17.5
feature_cols= [ 'PctBlack',  'Civilian_labor_force_2011']


--------------------------------------------------------------------------
## Regression Trees
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg


#Attepmt 1
feature_cols = ['serv', 'farm', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']

# list of values to try for max_depth
max_depth_range = range(1, 21)

# list to store the average RMSE for each value of max_depth
RMSE_scores = []

X = cnty_data_notnull[feature_cols]
y = cnty_data_notnull.MonthsReturn_pls1

# use 10-fold cross-validation with each value of max_depth
from sklearn.cross_validation import cross_val_score
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=10, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')
plt.savefig('DecisionTree1.png')

treereg = DecisionTreeRegressor(max_depth=6, random_state=1)
scores = cross_val_score(treereg, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

###RMSE = 19.9

#Attepmt 2
feature_cols = ['serv', 'farm','OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']

# list of values to try for max_depth
max_depth_range = range(1, 21)

# list to store the average RMSE for each value of max_depth
RMSE_scores = []

X = cnty_data_notnull[feature_cols]
y = cnty_data_notnull.MonthsReturn_pls1

# use 10-fold cross-validation with each value of max_depth
from sklearn.cross_validation import cross_val_score
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=10, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')

treereg = DecisionTreeRegressor(max_depth=6, random_state=1)
scores = cross_val_score(treereg, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

###RMSE = 19.9

##Attempt #3 (ALL)
feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']
# list of values to try for max_depth
max_depth_range = range(1, 21)

# list to store the average RMSE for each value of max_depth
RMSE_scores = []

X = cnty_data_notnull[feature_cols]
y = cnty_data_notnull.MonthsReturn_pls1

X.shape
y.shape
# use 10-fold cross-validation with each value of max_depth
from sklearn.cross_validation import cross_val_score
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X, y, cv=10, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')
plt.savefig('RegressionTree14.png')
treereg = DecisionTreeRegressor(max_depth=6, random_state=1)
scores = cross_val_score(treereg, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

##13.6


# create a Graphviz file
from sklearn.tree import export_graphviz
export_graphviz(treereg, out_file='tree_bikes.dot', feature_names=feature_cols)

# At the command line, run this to convert to PNG:
#   dot -Tpng tree_vehicles.dot -o tree_vehicles.png

## Try it with RANDOM FORESTS!

#Istantiate
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()
rfreg



#Attempt 1
feature_cols = ['serv', 'CoalPts', 'farm','OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']
X = cnty_data_notnull[feature_cols]
y = cnty_data_notnull.MonthsReturn_pls1

# list of values to try for n_estimators
#range --> Try every estimator from 10 to 310 by 10s
estimator_range = range(10, 310, 10)

# list to store the average RMSE for each value of n_estimators
RMSE_scores = []

# use 5-fold cross-validation with each value of n_estimators (WARNING: SLOW!)
for estimator in estimator_range:
    rfreg =  RandomForestRegressor(n_estimators=estimator, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot n_estimators (x-axis) versus RMSE (y-axis)
plt.plot(estimator_range, RMSE_scores)
plt.xlabel('n_estimators')
plt.ylabel('RMSE (lower is better)')
plt.savefig('RandomForest.png')

### RMSE = 17.6



#Attempt 2


feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']
X = cnty_data_notnull[feature_cols]
y = cnty_data_notnull.MonthsReturn_pls1

# list of values to try for n_estimators
#range --> Try every estimator from 10 to 310 by 10s
estimator_range = range(10, 310, 10)

# list to store the average RMSE for each value of n_estimators
RMSE_scores = []

# use 5-fold cross-validation with each value of n_estimators (WARNING: SLOW!)
for estimator in estimator_range:
    rfreg = RandomForestRegressor(n_estimators=estimator, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot n_estimators (x-axis) versus RMSE (y-axis)
plt.plot(estimator_range, RMSE_scores)
plt.xlabel('n_estimators')
plt.ylabel('RMSE (lower is better)')
plt.savefig('RandomForest.png')

## Pick a value around where it has leveled out

## RMSE = 11.6


# list of values to try for max_features
feature_range = range(1, len(feature_cols)+1)

# list to store the average RMSE for each value of max_features
RMSE_scores = []

# use 5-fold cross-validation with each value of max_features (WARNING: SLOW!)
for feature in feature_range:
    rfreg = RandomForestRegressor(n_estimators=120, max_features=feature, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

# plot max_features (x-axis) versus RMSE (y-axis)
plt.plot(feature_range, RMSE_scores)
plt.xlabel('max_features')
plt.ylabel('RMSE (lower is better)')


# max_features=30 is best and n_estimators=120 is sufficiently large
rfreg = RandomForestRegressor(n_estimators=120, max_features=30, oob_score=True, random_state=1)
rfreg.fit(X, y)

# compute feature importances
pd.DataFrame({'feature':feature_cols, 'importance':rfreg.feature_importances_}).sort('importance')


y_pred_rfreg_best = rfreg.predict(X)
y_pred_rfreg_best.shape
zip(y_pred_rfreg_best, cnty_data_notnull.series_id)


### Create a df of results, residuals, and map them
working = list(zip(y_pred_rfreg_best, cnty_data_notnull.series_id, cnty_data_notnull.LAT, cnty_data_notnull.LON, cnty_data_notnull.MonthsReturn_pls1))

working

list(working)

lol_tomap = list(working)
lol_tomap

headers = ('Ypred', 'Series_id', 'LAT', 'LON', 'MonthsToReturn')
RFRegResults = pd.DataFrame(working, columns=headers)

RFRegResults['Resid'] = RFRegResults.Ypred - RFRegResults.MonthsToReturn


##Map of RefRegResults
RFRegResults[ (RFRegResults.LON<0) & (RFRegResults.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Ypred', colormap='Reds')
plt.savefig('counties_RFReg_pred.png')

##Map of RefRegResults
RFRegResults[ (RFRegResults.LON<0) & (RFRegResults.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Resid', colormap='RdGy_r')
plt.savefig('counties_RFReg_resid.png')

Histogram of Residuals
RFRegResults.Resid.plot(kind='hist', bins=40)
plt.xlabel('Difference in Pred and True Months until return')
plt.ylabel('Number of Counties')
plt.savefig('counties_Return_predTrue_hist.png')



## box plot of months until return within 1%
RFRegResults.Resid.plot(kind='box')


RFRegResults.shape

RFRegResults.to_csv('RFRegResults.csv')   

-------------------------------------------------------------------------------

### Regression pt 2 ignore counties that returned at or before Jan 2010


## Try it with RANDOM FORESTS!

#Istantiate
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()
rfreg



#Attempt 1
feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']
X = cnty_data_notnull_rg2[feature_cols]
y = cnty_data_notnull_rg2.MonthsReturn_pls1

# list of values to try for n_estimators
#range --> Try every estimator from 10 to 310 by 10s
estimator_range = range(10, 310, 10)

# list to store the average RMSE for each value of n_estimators
RMSE_scores = []

# use 5-fold cross-validation with each value of n_estimators (WARNING: SLOW!)
for estimator in estimator_range:
    rfreg = RandomForestRegressor(n_estimators=estimator, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot n_estimators (x-axis) versus RMSE (y-axis)
plt.plot(estimator_range, RMSE_scores)
plt.xlabel('n_estimators')
plt.ylabel('RMSE (lower is better)')
plt.savefig('RandomForests_RG2.png')

## RMSE = 11.65


### Linear regression:

from sklearn.cross_validation import train_test_split


# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(feature_cols):
    X = cnty_data_notnull_rg2[feature_cols]
    y = cnty_data_notnull_rg2.MonthsReturn_pls1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

## Attempt 1
feature_cols = ['serv', 'farm', 'HolidaySeason']
train_test_rmse(feature_cols)
# RMSE = 15.7688
linreg.coef_
## Attempt 2
feature_cols= ['LAT', 'LON', 'CoalPts', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus']
train_test_rmse(feature_cols)
# RMSE = 13.453
linreg.coef_


----------------------------------------------------------------------------
###Classification modeling --- Logistic Regression


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
    return metrics.confusion_matrix(y_test, y_pred_class)

    
def train_test_accuracy_score(feature_cols):
    X = cnty_data_notnull_cl[feature_cols]
    y = cnty_data_notnull_cl.Returned
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    logreg = LogisticRegression(C=1e9)
    logreg.fit(X_train, y_train)
    y_pred_class = logreg.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred_class)

def roc_curve_reg(feature_cols):
    X = cnty_data_notnull_cl[feature_cols]
    y = cnty_data_notnull_cl.Returned
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    logreg = LogisticRegression(C=1e9)
    logreg.fit(X_train, y_train)
    y_pred_class = logreg.predict(X_test)
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]
    # plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    return plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
  
### NULL ACCURACY
# this works regardless of the number of classes
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
y_test.value_counts().head(1) / len(y_test)

### Null Accuracy = .878788


# Attempt 1
feature_cols = ['serv', 'farm', 'LAT', 'LON', 'Business_Estab_20to99_emp', 'poploss','PctBlack', 'OilRegion', 'coalBinay', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']
train_test_auc_score(feature_cols)
train_test_confu_matrix(feature_cols)
train_test_accuracy_score(feature_cols)
### AUC = .739
###Accuracy = .872
### Confusion Matrix
#array([[  1,  91],
#       [  6, 661]])

confusion= train_test_confu_matrix(feature_cols)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)



## Print a roc curve

 X = cnty_data_notnull_cl[feature_cols]
 y = cnty_data_notnull_cl.Returned
 X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
 logreg = LogisticRegression(C=1e9)
 logreg.fit(X_train, y_train)
 y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


## Print out histogram of results
import matplotlib.pyplot as plt
plt.hist(y_pred_prob)
plt.xlim(0, 1)
plt.xlabel('Predicted probability of returning')
plt.ylabel('Frequency')
plt.savefig('LogReg_HistogramOfPredProb.png')

### increase the threshold
y_pred_class = np.where(y_pred_prob > 0.75, 1, 0)
metrics.confusion_matrix(y_test, y_pred_class)
metrics.roc_auc_score(y_test, y_pred_prob)
metrics.accuracy_score(y_test, y_pred_class)
#array([[ 29,  63],
#      [ 52, 615]]



# Attempt 2
feature_cols = ['serv', 'farm', 'LAT', 'LON', 'OilRegion', 'coalBinay']
train_test_auc_score(feature_cols)
train_test_confu_matrix(feature_cols)
train_test_accuracy_score(feature_cols)
### AUC = .75
## Accuracy = .8787878

confusion = train_test_confu_matrix(feature_cols)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)


'''True Positives: 667
True Negatives: 0
False Positives: 92
False Negatives: 0'''

# Attempt 3 (All appropriate features)
feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'StimPC_Limit', 'StimPC_Limit2', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']
train_test_auc_score(feature_cols)
train_test_confu_matrix(feature_cols)
train_test_accuracy_score(feature_cols)

### AUC = .58966
## Accuracy = .8787878

confusion = train_test_confu_matrix(feature_cols)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)


'''True Positives: 667
True Negatives: 0
False Positives: 92
False Negatives: 0'''



# Attempt 4 (All appropriate features)
feature_cols= ['Average2007']
train_test_auc_score(feature_cols)
train_test_confu_matrix(feature_cols)
train_test_accuracy_score(feature_cols)

## Print out histogram of results
import matplotlib.pyplot as plt
plt.hist(y_pred_prob)
plt.xlim(0, 1)
plt.xlabel('Predicted probability of returning')
plt.ylabel('Frequency')
plt.savefig('LogReg_HistogramOfPredProb.png')

### increase the threshold
y_pred_class = np.where(y_pred_prob > 0.75, 1, 0)
metrics.confusion_matrix(y_test, y_pred_class)
#array([[ 29,  63],
#      [ 52, 615]]

## couldn't get it above null accuracy, think i'm missing something important..
-------------------
### Try it with knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

## Attempt 1
feature_cols = ['serv', 'farm', 'LAT', 'LON', 'Business_Estab_20to99_emp', 'poploss','PctBlack', 'OilRegion', 'coalBinay', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=25)
type(knn)

knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
# compute classification accuracy

metrics.accuracy_score(y_test, y_pred_class)
## 88% Accuracy
metrics.roc_auc_score(y_test, y_pred_class)
## AUC = .505

confusion = metrics.confusion_matrix(y_test,y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''True Positives: 667
True Negatives: 1
False Positives: 91
False Negatives: 0 '''

### Again, I am having a very hard time with fasle positives

## Attempt 2

feature_cols = ['serv', 'farm', 'LAT', 'LON', 'OilRegion', 'coalBinay']
## Define x and y
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# make an instance of a KNeighborsClassifier object with n_neighbors
knn = KNeighborsClassifier(n_neighbors=13)

knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
# compute classification accuracy

metrics.accuracy_score(y_test, y_pred_class)
## .8998 Accuracy

metrics.roc_auc_score(y_test, y_pred_class)
## .6759 AUC

confusion = metrics.confusion_matrix(y_test,y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''True Positives: 667
True Negatives: 0
False Positives: 92
False Negatives: 0'''


#Attempt 3 (ALL!)
feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'StimPC_Limit', 'StimPC_Limit2', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']

X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=12)
type(knn)

knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
# compute classification accuracy

metrics.accuracy_score(y_test, y_pred_class)
metrics.roc_auc_score(y_test, y_pred_class)

##  .87 Accuracy
## .5 AUC

confusion = metrics.confusion_matrix(y_test,y_pred_class)
confusion

## Try it with classification trees
from sklearn.tree import DecisionTreeClassifier

##Attempt 1
feature_cols = ['serv', 'farm', 'LAT', 'LON', 'Business_Estab_20to99_emp', 'poploss','PctBlack', 'OilRegion', 'coalBinay', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013']

X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

treeclf = DecisionTreeClassifier(max_depth=10, random_state=1)
treeclf.fit(X_train, y_train)
y_pred_class = treeclf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)
metrics.roc_auc_score(y_test, y_pred_class)
### Accuracy.869
### AUC = .677

confusion = metrics.confusion_matrix(y_test,y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''True Positives: 606
True Negatives: 40
False Positives: 52
False Negatives: 6'''

# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})


##Attempt 2
feature_cols = ['serv', 'farm', 'LAT', 'LON', 'OilRegion', 'coalBinay']
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
treeclf.fit(X_train, y_train)
y_pred_class = treeclf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)
metrics.roc_auc_score(y_test, y_pred_class)
###Accuracy .888
### Auc .5755


confusion = metrics.confusion_matrix(y_test,y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''True Positives: 659
True Negatives: 15
False Positives: 77
False Negatives: 8'''

# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})


# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})



###Try 3 (No Seasonal)

feature_cols= ['LAT', 'LON', 'CoalPts', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'StimPC_Limit', 'StimPC_Limit2', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'coalBinay']
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

treeclf = DecisionTreeClassifier(max_depth=5, random_state=1)
treeclf.fit(X_train, y_train)
y_pred_class = treeclf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

###.87777

confusion = metrics.confusion_matrix(y_test,y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''True Positives: 653
True Negatives: 13
False Positives: 79
False Negatives: 14'''




# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})



### Try Naive Bayes
# import both Multinomial and Gaussian Naive Bayes
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics


#TRY1
feature_cols= ['LAT','CoalPts', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013',  'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'StimPC_Limit', 'StimPC_Limit2', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus']

# create X and y
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_class = mnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)


# testing accuracy of Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class = gnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

## Gausian NB gave .16 Accuracy Score for my 2 classess (rs=1), and again for rs=15

confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''True Positives: 32
True Negatives: 90
False Positives: 2
False Negatives: 635'''

##Try 2
feature_cols = ['serv', 'farm', 'LAT', 'LON', 'OilRegion', 'coalBinay']

# create X and y
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_class = mnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)


# testing accuracy of Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class = gnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

## Gausian NB gave .86 Accuracy Score for my 2 classess (rs=1), and again for rs=15

confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''True Positives: 646
True Negatives: 9
False Positives: 83
False Negatives: 2'''

##Try 3
feature_cols= ['LAT', 'LON', 'CoalPts', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'StimPC_Limit', 'StimPC_Limit2', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'coalBinay']


# create X and y
X = cnty_data_notnull_cl[feature_cols]
y = cnty_data_notnull_cl.Returned

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_class = mnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)


# testing accuracy of Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class = gnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

## Gausian NB gave .16 Accuracy Score for my 2 classess (rs=1), and again for rs=15

confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

'''
True Positives: 32
True Negatives: 90
False Positives: 2
False Negatives: 635'''



-------------------------------------------------------------------------------


----------------------------------------------------------------------------
###Classification modeling  with 3 classess 
###--- KNN

### NULL ACCURACY
# this works regardless of the number of classes
X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl2.ReturnClass2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
y_test.value_counts().head(1) / len(y_test)

### Null Accuracy = .816
KNN

### Try it with knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

## Attempt 1
feature_cols = ['serv', 'coalBinay', 'OilRegion', 'farm', 'LAT', 'LON', 'poploss', 'PctBlack', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013'] 

X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl2.ReturnClass2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=25)
type(knn)

knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
# compute classification accuracy

metrics.accuracy_score(y_test, y_pred_class)

## .814 Accuracy

## Attempt 2

feature_cols = ['serv', 'farm', 'HolidaySeason', 'LAT', 'LON', 'poploss', 'PctBlack', 'Average2007', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013', 'January','February', 'April', 'May', 'October', 'November','CoalPts']

X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl2.ReturnClass2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=30)
type(knn)

knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
# compute classification accuracy

metrics.accuracy_score(y_test, y_pred_class)

## .816 Accuracy

y_test.shape
y_pred_class.shape
metrics.confusion_matrix(y_test,y_pred_class)

'''array([[  1,   0,  91],
       [  0,   0,  47],
       [  1,   0, 619]]'''
       
       
#Attempt 3 (ALL!)
feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'Summer678', 'Spring345', 'Fall91011', 'StimPC_Limit', 'StimPC_Limit2', 'HolidaySeason', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus', 'CoalPts', 'coalBinay']

X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl2.ReturnClass2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=8)
type(knn)

knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
# compute classification accuracy

metrics.accuracy_score(y_test, y_pred_class)

## .818 Accuracy


## Try it with classification trees
from sklearn.tree import DecisionTreeClassifier

##Attempt 1
feature_cols = ['serv', 'OilRegion', 'farm', 'poploss', 'PctBlack', 'Pct_SomeCollege_2009_2013', 'StimPC_Limit', 'Median_Household_Income_2013'] 

X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl.ReturnClass2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

treeclf = DecisionTreeClassifier(max_depth=5, random_state=1)
treeclf.fit(X_train, y_train)
y_pred_class = treeclf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

###.814


# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})

##Attempt 2 (All)
feature_cols= ['LAT', 'LON', 'CoalPts', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'Civilian_labor_force_2011', 'Median_Household_Income_2013', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_SomeCollege_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'manf', 'fsgov', 'serv', 'nonsp', 'house', 'loweduc', 'lowemp', 'perpov', 'poploss', 'rec', 'retire', 'perchldpov', 'PctBlack', 'StimPC_Limit', 'StimPC_Limit2', 'OilRegion', 'Bakken', 'EagleFord', 'Marcellus']
X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl2.ReturnClass2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

treeclf = DecisionTreeClassifier(max_depth=5, random_state=1)
treeclf.fit(X_train, y_train)
y_pred_class = treeclf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

###.819

confusion = metrics.confusion_matrix(y_test,y_pred_class)
confusion
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]


# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})



###Try 3 (No Seasonal)

feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013','Civilian_labor_force_2011', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'lowemp',  'poploss',  'StimPC_Limit2', 'OilRegion', 'coalBinay']
X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl2.ReturnClass2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
treeclf.fit(X_train, y_train)
y_pred_class = treeclf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

###.823


# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})



### Try Naive Bayes
# import both Multinomial and Gaussian Naive Bayes
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics


#TRY1
feature_cols= ['LAT', 'LON', 'Rural_urban_continuum_code_2013','Civilian_labor_force_2011', 'Median_Household_Income_Percent_of_State_Total_2013', 'Pct_less_than_hs_2009_2013', 'Pct_HS_Grad_only_2009_2013', 'Pct_college_grad_2009_2013', 'Net_Mig_2010_2014', 'MigRate_2010_2014', 'Business_Estab_under_20_emp', 'Business_Estab_20to99_emp', 'Business_Estab_100to499_emp', 'Business_Estab_over500_emp', 'Stimulus', 'StimulusPerCapita', 'farm', 'mine', 'lowemp',  'poploss',  'StimPC_Limit2', 'OilRegion', 'coalBinay']

# create X and y
X = cnty_data_notnull_cl2[feature_cols]
y = cnty_data_notnull_cl2.ReturnClass2

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15)

# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_class = mnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)


# testing accuracy of Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class = gnb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)

## Gausian NB gave .50 Accuracy Score for my 3 classess (rs=1), and again for rs=15



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

# Join County Names

cnty_data = pd.read_csv('Cty_Results_v8.csv', header=0)

cnty_data.head()
cnty_data.columns
cnty_data.dtypes
## Plot histogram of months til return within 1%
cnty_data[cnty_data.MonthsReturn_pls1 > 0].MonthsReturn_pls1.plot(kind='hist', bins=80)
plt.xlabel('Months until return within 1% of 2007 Ave Unemployment Rate')
plt.ylabel('Number of Counties')
plt.savefig('counties_Return_hist.png')
## box plot of months until return within 1%
cnty_data[cnty_data.MonthsReturn_pls1 > 0].MonthsReturn_pls1.plot(kind='box')
## plot fips codes vs months (quick spatial look)

cnty_data.dtypes
cnty_data[cnty_data.MonthsReturn_pls1 > 0].plot(kind='scatter', x='Pct_less_than_hs_2009_2013', y='MonthsReturn_pls1')
import seaborn as sns
cnty_data[cnty_data.MonthsReturn_pls1 > 0].plot(kind='scatter', x='PctBlack', y='MonthsReturn_pls1')

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

##Map of 
cnty_data[(cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Urban_influence_code_2013', colormap='Reds')
plt.savefig('counties_map_Rural_Urban.png')

cnty_data[(cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Rural_urban_continuum_code_2013', colormap='Reds')
plt.savefig('counties_map_Rural_Urban.png')


cnty_data[(cnty_data.LON<0) & (cnty_data.LON>-130)].plot(kind='scatter', x='LON', y='LAT', c='Rural_urban_continuum_code_2013', colormap='Reds')
plt.savefig('counties_map_Rural_Urban.png')

cnty_data[cnty_data.MonthsReturn_pls1 > 0].MonthsReturn_pls1.value_counts().plot(kind='bar')

placeNms.head()
placeNms.rename(columns={'FIPS':'FIPS_Code'}, inplace=True)

cnty_data_wNames = pd.merge(cnty_data, placeNms, how='inner')
cnty_data_wNames.head


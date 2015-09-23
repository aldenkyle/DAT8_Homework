# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:33:35 2015

@author: kylealden
"""

url = 'https://raw.githubusercontent.com/justmarkham/DAT4-students/master/kerry/Final/NBA_players_2015.csv'
bball = pd.read_csv(url, header=0)
bball.head()

bball.columns

bball['pos_num'] = bball.pos.map({'F':0, 'G':1, 'C':2})


X = bball[["ast", "stl", 'blk', 'tov', 'pf']]
X2 = bball[["ast", "stl", 'blk', 'pf', 'tov']]
y = bball.pos_num


from sklearn.neighbors import KNeighborsClassifier


## KNN with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X2, y)

knn.predict([1, 1, 0, 1,2])

## Probably a guard
X_new = [1,1,0,1,2]

knn.predict_proba(X_new)

### Probabilities 
### array([[ 0.2,  0.8,  0. ]])



## KNN with 50 neighbors

knn = KNeighborsClassifier(n_neighbors=50)

knn.fit(X, y)

knn.predict([1, 1, 0, 1,2])
## probably a forward
X_new = [1,1,0,1,2]

knn.predict_proba(X_new)
##array([[ 0.62,  0.32,  0.06]])


bball.boxplot(column='ast', by='pos')
bball.boxplot(column='stl', by='pos')
bball.boxplot(column='tov', by='pos')
bball.boxplot(column='blk', by='pos')
bball.boxplot(column='pf', by='pos')


## assists are good for telling guards
## blocks are great for preicting centers

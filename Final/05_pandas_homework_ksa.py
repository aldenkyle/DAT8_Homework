'''
Pandas Homework with IMDb data
'''

'''
BASIC LEVEL
'''

import pandas as pd
import matplotlib.pyplot as plt

# read in 'imdb_1000.csv' and store it in a DataFrame named movies
movies = pd.read_table('imdb_1000.csv', sep = ",")
movies.isnull().sum()

# check the number of rows and columns

movies.shape
# check the data type of each column
movies.dtypes
# calculate the average movie duration
movies.duration.mean()
'''The average movie duration was 120.98 minutes'''

# sort the DataFrame by duration to find the shortest and longest movies
movies.sort("duration").head(1)
movies.sort("duration").tail(1)
'''The shortest movie in the dataset is "Freaks" which was 64 minutes.The 
longest move in the dataset is "Hamlet" which was 224 minutes long.'''
# create a histogram of duration, choosing an "appropriate" number of bins
movies.duration.plot(kind='hist', bins=10, title='Duration of Movies in IMDB Dataset')
plt.xlabel('Duration in Minutes')
plt.ylabel('Number of Movies')

# use a box plot to display that same data

movies.duration.plot(kind='box', title='Duration of Movies in IMDB Dataset')

'''
INTERMEDIATE LEVEL
'''

# count how many movies have each of the content ratings
movies.content_rating.value_counts()

''' Result:
R            460
PG-13        189
PG           123
NOT RATED     65
APPROVED      47
UNRATED       38
G             32
PASSED         7
NC-17          7
X              4
GP             3
TV-MA          1 '''

# use a visualization to display that same data, including a title and x and y labels
movies.content_rating.value_counts().plot(kind='bar', title='Content Rating of Movies in IMDB Dataset')
plt.xlabel('Content Rating')
plt.ylabel('Number of Movies')

# convert the following content ratings to "UNRATED": NOT RATED, APPROVED, PASSED, GP

movies.content_rating.replace(['NOT RATED','APPROVED','PASSED','GP'], "UNRATED", inplace=True)

# convert the following content ratings to "NC-17": X, TV-MA

movies.content_rating.replace(['X','TV-MA'], "NC-17", inplace=True)

# count the number of missing values in each column

movies.isnull().sum()
 ###he only column with missing values is content_rating which has 3

# if there are missing values: examine them, then fill them in with "reasonable" values
movies.content_rating.fillna(value='UNRATED', inplace=True)  
# calculate the average star rating for movies 2 hours or longer,
movies[movies.duration >= 120].star_rating.mean()

'''The average star rating for movies 2 hours or longer is 7.95'''

# and compare that with the average star rating for movies shorter than 2 hours
movies[movies.duration < 120].star_rating.mean()

'''The average star rating for movies shorter than 2 hours is 7.84'''
# use a visualization to detect whether there is a relationship between duration and star rating

movies.plot(kind='scatter', x='duration', y='star_rating')

 ### Based on the scatter plot i do not detect a clear relationship between
 ### and duration, but the might be a slight trend toward better rating 
 ### for longer movies. 



# calculate the average duration for each genre

movies.groupby('genre').duration.mean().sort()


''' Average Duration by Genre
Action       126.485294
Adventure    134.840000
Animation     96.596774
Biography    131.844156
Comedy       107.602564
Crime        122.298387
Drama        126.539568
Family       107.500000
Fantasy      112.000000
Film-Noir     97.333333
History       66.000000
Horror       102.517241
Mystery      115.625000
Sci-Fi       109.000000
Thriller     114.200000
Western      136.666667'''

'''
ADVANCED LEVEL
'''

# visualize the relationship between content rating and duration

movies.boxplot(column='duration', by='content_rating')

 ### G rated movies tend to be shorter, while PG-13 and R rated movies 
 ### tend to be the longer. 

# determine the top rated movie (by star rating) for each genre

idx = movies.groupby(['genre'])['star_rating'].transform(max) == movies['star_rating']

movies[idx].sort('genre').drop(['content_rating', 'duration', 'actors_list'], axis=1)

 ### Wasn't sure how to do this, found this solution on stack overflow using transform...
'''
     star_rating                                          title      genre
3            9.0                                The Dark Knight     Action
7            8.9  The Lord of the Rings: The Return of the King  Adventure
30           8.6                                  Spirited Away  Animation
8            8.9                               Schindler's List  Biography
25           8.6                              Life Is Beautiful     Comedy
35           8.6                                   Modern Times     Comedy
29           8.6                                    City Lights     Comedy
0            9.3                       The Shawshank Redemption      Crime
9            8.9                                     Fight Club      Drama
5            8.9                                   12 Angry Men      Drama
468          7.9                     E.T. the Extra-Terrestrial     Family
638          7.7                      The City of Lost Children    Fantasy
105          8.3                                  The Third Man  Film-Noir
338          8.0                            Battleship Potemkin    History
39           8.6                                         Psycho     Horror
38           8.6                                    Rear Window    Mystery
145          8.2                                   Blade Runner     Sci-Fi
350          8.0                              Shadow of a Doubt   Thriller
6            8.9                 The Good, the Bad and the Ugly    Western'''

# check if there are multiple movies with the same title, and if so, determine if they are actually duplicates

movies.title.duplicated().sum()
movies[movies.title.duplicated()]

movies.duplicated().sum()


movies[(movies.title == "Dracula") | (movies.title == "True Grit") | (movies.title == "Les Miserables") | (movies.title == "The Girl with the Dragon Tattoo") ]

 ### There are 4 movies with duplicate titles, "The Girl with the Dragon
 ### Tatoo", "Dracula", "Les Miserables", and "True Grit". None of them are
 ### as evidenced by movies.duplicated.sum() which showed that there are 0
 ### rows that are exact duplicates. I also printed out the dataframe for those
 ### 4 movies just to check. 



# calculate the average star rating for each genre, but only include genres with at least 10 movies

movies.groupby("genre").star_rating.agg(['count', 'mean']).sort('count').tail(9)

 ###Again, wasn't sure how to accomplish this logic without knowing how many genres
 ### were over 10. I looked around and found a filter method but couldn't get it to work.
'''           count      mean
genre                     
Mystery       16  7.975000
Horror        29  7.806897
Animation     62  7.914516
Adventure     75  7.933333
Biography     77  7.862338
Crime        124  7.916935
Action       136  7.884559
Comedy       156  7.822436
Drama        278  7.902518'''

'''
BONUS
'''

# Figure out something "interesting" using the actors data!

# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords

# Importing data files
name_basics = pd.read_csv('name_basics.tsv',sep='\t')
title_basics = pd.read_csv('title_basics.tsv',sep='\t')
title_ratings = pd.read_csv('title_ratings.tsv',sep = '\t')
title_crew = pd.read_csv('title_crew.tsv',sep = '\t')

# Preprocessing data

# Replacing "\N" with NaN for better cleaning
name_basics.replace(to_replace = r'\N', value = np.nan, inplace = True)
title_basics.replace(to_replace = r'\N', value = np.nan, inplace = True)
title_ratings.replace(to_replace = r'\N', value = np.nan, inplace = True)
title_crew.replace(to_replace = r'\N', value = np.nan, inplace = True)

# Removing Adult titles
title_basics = title_basics[title_basics['isAdult']==0]

# Merging title metadata and ratings and dropping irrelevant columns
title = pd.merge(title_basics,title_ratings,on='tconst')
title.drop(['originalTitle','isAdult','endYear','runtimeMinutes'],axis=1,inplace=True)
name_basics.drop(['birthYear','deathYear'],axis=1,inplace=True)

# Dropping Nan values from "startYear" and "genres" columns and converting startYear values to int
title.dropna(subset = ['startYear','genres'],inplace = True)
title['startYear'] = title['startYear'].astype(int)

# Merging crew dataset with the combined one
title = pd.merge(title,title_crew,on='tconst')

# Creating a new feature called "Popularity" which will help in filtering titles which would be relevant to the users
title['Popularity'] = title['averageRating']/title['averageRating'].mean()+title['numVotes']/title['numVotes'].mean()

title = title[title['Popularity']>=title['Popularity'].mean()]

# Removing tvEpisodes and shorts entries
title = title[title['titleType'] != 'tvEpisode']
title = title[title['titleType'] != 'short']

# Creating bag of words

# Removing entries with Nan values in "directors" and "writers" columns
title.dropna(subset=['directors','writers'],inplace=True)

# Importing stopwords from the nltk library
stop = stopwords.words('english')

# Converting title names to lower-case
title['primaryTitle'] = title['primaryTitle'].str.lower()

# Removing stopwords from title names
title['primaryTitle'] = title['primaryTitle'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Combining title names, genres, directors and writers to create a bag of words
title['tags']=title['primaryTitle'].map(str)+','+title['genres']+','+title['directors'].map(str)+','+title['writers']

# Dropping columns whose values have been added to the bag of words column
title.drop(['primaryTitle','genres','directors','writers'],axis=1,inplace=True)

#----Preprocessing end----
import pandas as pd
import numpy as np
import seaborn as sns
import rec_Sys_main as get_rec
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
    
stop = stopwords.words('english')    
name_basics = pd.read_csv('name_basics.tsv',sep='\t')
title_basics = pd.read_csv('title_basics.tsv',sep='\t')
title_ratings = pd.read_csv('title_ratings.tsv',sep = '\t')
title_crew = pd.read_csv('title_crew.tsv',sep = '\t')

# Preprocessing data
name_basics.replace(to_replace = r'\N', value = np.nan, inplace = True)
title_basics.replace(to_replace = r'\N', value = np.nan, inplace = True)
title_ratings.replace(to_replace = r'\N', value = np.nan, inplace = True)
title_crew.replace(to_replace = r'\N', value = np.nan, inplace = True)

title_basics = title_basics[title_basics['isAdult']==0]
title = pd.merge(title_basics,title_ratings,on='tconst')
title.drop(['originalTitle','isAdult','endYear','runtimeMinutes'],axis=1,inplace=True)
name_basics.drop(['birthYear','deathYear'],axis=1,inplace=True)

title.dropna(subset = ['startYear','genres'],inplace = True)

title = pd.merge(title,title_crew,on='tconst')

title['startYear'] = title['startYear'].astype(int)

title['Popularity'] = title['averageRating']/title['averageRating'].mean()+title['numVotes']/title['numVotes'].mean()

title = title[title['Popularity']>=title['Popularity'].mean()]

title = title[title['titleType'] != 'tvEpisode']
title = title[title['titleType'] != 'short']

# Creating bag of words
title.dropna(subset=['directors','writers'],inplace=True)

#stop = stopwords.words('english')

title['primaryTitle'] = title['primaryTitle'].str.lower()
title['primaryTitle'] = title['primaryTitle'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

title['tags']=title['primaryTitle'].map(str)+','+title['genres']+','+title['directors'].map(str)+','+title['writers']

title.drop(['primaryTitle','genres','directors','writers'],axis=1,inplace=True)

# Visualisation
"""title['numVotes'].mean()+title['numVotes'].std()
title['averageRating'].mean()
sns.distplot(title['numVotes'],hist=False)"""

title.to_csv('newframe.tsv',index=False,sep='\t')
print('-----------------Preprocessing done------------------')


test = input('Enter tconst of title: ')
get_rec.main(test)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = None
import rec_Sys_postprocessing as post
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def main(test_title):
    print('----------------Running the recommendation engine-----------------')
    df = pd.read_csv('newframe.tsv',sep='\t')
    title_basics = pd.read_csv('title_basics.tsv',sep='\t')
    new_df = df[(df.averageRating >7.5) & (df.startYear >2000)]
    
    #Computing the TF_IDF matrix based on the term frequency in the 'tags' column
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(new_df['tags'])
    T = tfidf_matrix
    
    #Similarity matrix using the dot product
    cosine_similarity_matrix = linear_kernel(T,T)
    
    #Indexing the new_dataframe to accesses the relavant Titles with their IDs
    new_df = new_df.reset_index()
    titles = new_df['tconst']
    indices = pd.Series(new_df.index, index = new_df['tconst'])
    
    #Input: title id
    #Output: 10 similar movies based on the similarity scores
    def recommend(title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]
    
    recommendations = recommend(test_title)
    
    # Creating lookup table for post_processing
    title_lookup = title_basics.loc[title_basics['tconst'].isin(new_df['tconst'])]
    title_lookup.drop(['originalTitle','isAdult','endYear','runtimeMinutes'],axis=1,inplace=True)
    
    title_lookup.to_csv('title_lookup.tsv',index=False,sep='\t')
    
    post.post_process(recommendations)
# Multi-Criteria-Recommender-System
A project to build a multi-criteria recommender system, which works on movies, TV shows,  Video Games and other multimedia titles.

# Data and Features
The dataset used is the complete [IMDB dataset](https://datasets.imdbws.com/) which is free to use and very comprehensive.

Features used are Genre, Year of release, Writers and Directors, and Popularity, which is basically a constructed function of Number of Votes and Average Rating. Because these features are common across movies, TV shows, video games and others, a multi-criteria model can be created in a relatively straightforward way.

# Bag-of-words Model
A TfIdf vectorizer was used to construct a TfIdf matrix, using unigrams and bigrams.
Recommendations were made on the basis of cosine similarity.

# References

* https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243

* https://www.imdb.com/interfaces/

* https://github.com/jaypatel00174/Movie-Recommendations/blob/master/movies_recsys.ipynb

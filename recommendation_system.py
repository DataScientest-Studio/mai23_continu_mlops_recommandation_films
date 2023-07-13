import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split

# ouverture des fichiers pkl
collab_filtering = pd.read_pickle('collab_filtering.pkl')
content_based_filtering_bis = pd.read_pickle('content_based_filtering.pkl')
dict_person = pd.read_pickle('dict_person.pkl')
dict_movie = pd.read_pickle('dict_movie.pkl')

#############################################################################Content_based_filtering##################################################################################

# supprimer les doublons
content_based_filtering_bis_duplicated = content_based_filtering_bis.iloc[:,1:].drop_duplicates(keep = 'last')

#standardiser les colonnes averageRating, startYear, runtimeMinutes
scaler = MinMaxScaler()
df_scaler = pd.DataFrame(scaler.fit_transform(content_based_filtering_bis_duplicated.iloc[:,[1,5,6]]), 
                         columns= content_based_filtering_bis_duplicated.iloc[:,[1,5,6]].columns)

# création d'un dictionnaire movie : movieIndex
content_based_filtering_bis_duplicated.reset_index(inplace= True)
movies = content_based_filtering_bis_duplicated['movieId'].apply(lambda x : dict_movie[x])
movies_index = pd.Series(movies.index, index = movies)

# TfidVectorizer pour transformer les colonnes textuelles en vecteurs numériques se basant sur les mots et leur fréquence d'apparition
tfid = TfidfVectorizer(stop_words='english')
tfid_genres = tfid.fit_transform(content_based_filtering_bis_duplicated['genres_y'])
tfid_directors = tfid.fit_transform(content_based_filtering_bis_duplicated['directors'])
tfid_writers = tfid.fit_transform(content_based_filtering_bis_duplicated['writers'])

# concaténer le df
df_concat = pd.concat([df_scaler,
                       pd.DataFrame.sparse.from_spmatrix(tfid_genres),
                       pd.DataFrame.sparse.from_spmatrix(tfid_directors),
                       pd.DataFrame.sparse.from_spmatrix(tfid_writers)], axis = 1)

# réduire la taille du df
df_concat = df_concat.astype('float16')

#Nearest Neighbors
model_knn = NearestNeighbors(n_neighbors=21)
model_knn.fit(df_concat)

kneighbors_50 = model_knn.kneighbors(np.array(df_concat),51,return_distance=True)

def recommendation_movies_knn(movie):  
    """
    movie : nom du film
    retourne une liste de 20 films les plus similaires au film rentré en paramètre
    """
    return movies.iloc[kneighbors_50[1][movies_index[movie]][1:21]]

def recommendation_movies_knn_user(movie,userId):   
    """
    movie : nom du film
    userId : identifiant de l'utilisateur
    retourne une liste de 20 films les plus similaires au film rentré en paramètre et non déjà vus par l'utilisateur en 
    question
    """
    return movies.iloc[kneighbors_50[1][movies_index[movie]]][~movies.iloc[kneighbors_50[1][movies_index[movie]]].index.isin(collab_filtering[collab_filtering['userId'] == userId]['movieId'])][1:]


#############################################################################Collaborative_filtering##################################################################################

def collaborative_filtering(userId,n_recommendation,svd_model):
    """
    userId : identifiant de l'utilisateur ;
    n_recommendation : nombre de recommandation ;
    svd_model : algorithme de SVD ;
    retourne une liste de n films recommandés non déja vus par cet userId se basant sur les notes mises sur une liste de films
    et l'interaction avec d'autres utilisateurs et leurs notes
    """
    
    global collab_filtering
    collab_filtering_df = collab_filtering
    collab_filtering_df = collab_filtering_df.drop_duplicates(subset = ['movieId'],keep = 'first')
    collab_filtering_df['est'] = collab_filtering_df.apply(lambda x: svd_model.predict(userId,x['movieId'], x['rating']).est,
                                                           axis = 1)
    collab_filtering_df = collab_filtering_df.iloc[:,[1,3]]
    
    movies_sorted = collab_filtering_df.sort_values(['est'], ascending=False).merge(right = pd.DataFrame(content_based_filtering_bis_duplicated[content_based_filtering_bis_duplicated['movieId'].isin(collab_filtering_df['movieId'])]['movieId']).reset_index(),
                                                                                 on = 'movieId',
                                                                                 how = 'inner')
    movies_sorted = movies_sorted[~movies_sorted['movieId'].isin(collab_filtering[collab_filtering['userId'] == userId]['movieId'].unique())]
    return movies[movies_sorted['index']][0:n_recommendation]

# surprise
reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(collab_filtering,reader = reader)

# séparer les données en un jeu de test et un jeu d'entraînement
trainset, testset = train_test_split(data, test_size=0.25,random_state=10)

# entraînement du modèle
svd = SVD()
svd.fit(trainset)

# tester sur le jeu de test
test_pred = svd.test(testset)

# évaluation du model sur le jeu de test
accuracy.rmse(test_pred, verbose=True) 

#############################################################################Hybrid_recommandation_filtering##################################################################################

def hybrid_recommendation_movies(userId,movie,svd_model):
    """
    userId : identifiant de l'utilisateur ;
    movie : nom du film ; 
    svd_model : algorithme de SVD ;
    retourne une liste de n films recommandés à partir d'un film se basant à la fois sur les similitudes entre les films 
    mais également sur les interactions entre les notes des différents utilisateurs (système hybrid)
    le nombre de recommandation dépend de celui fait dans le système based_content
    """
    ratings_from_content_filtering = collab_filtering[collab_filtering['movieId'].isin(collab_filtering[collab_filtering.index.isin(pd.Series(content_based_filtering_bis_duplicated['index'],
                                                                                                         index = movies_index)[recommendation_movies_knn_user(movie,userId).index])]['movieId'].unique())]
    
   
    ratings_from_content_filtering['est'] = ratings_from_content_filtering.apply(lambda x: svd_model.predict(userId,x['movieId'], x['rating']).est, axis = 1)
    
    ratings_from_content_filtering = ratings_from_content_filtering.iloc[:,[1,3]].drop_duplicates(keep = 'first')
    
    movies_sorted = ratings_from_content_filtering.sort_values(['est'], ascending=False).merge(right = pd.DataFrame(content_based_filtering_bis_duplicated[content_based_filtering_bis_duplicated['movieId'].isin(ratings_from_content_filtering['movieId'])]['movieId']).reset_index(),
                                                                                 on = 'movieId',
                                                                                 how = 'inner')
    
    return movies[movies_sorted['index']][:20]

# surprise
reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(collab_filtering,reader = reader)

# séparer les données en un jeu de test et un jeu d'entraînement
trainset, testset = train_test_split(data, test_size=0.25,random_state=10)

# entraînement du modèle
svd = SVD()
svd.fit(trainset)

# tester sur le jeu de test
test_pred = svd.test(testset)

# évaluation du model sur le jeu de test
accuracy.rmse(test_pred, verbose=True) 

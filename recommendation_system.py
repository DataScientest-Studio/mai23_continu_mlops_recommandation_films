import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split

# ouverture des fichiers
def open_tsv_data(tsv_file):
    return pd.read_csv('C:/Users/antho/Projet_recommandation_system/{}.tsv.gz'.format(tsv_file), compression='gzip',sep = '\t', na_values = '\\N')

def open_csv_data(csv_file):
    return pd.read_csv('C:/Users/antho/Projet_recommandation_system/{}.csv'.format(csv_file))

title_ratings = open_tsv_data(tsv_file='title.ratings')
title_crew = open_tsv_data(tsv_file='title.crew')
title_basics = open_tsv_data(tsv_file='title.basics')
title_principals = open_tsv_data(tsv_file='title.principals')

ratings = open_csv_data('ratings')
links = open_csv_data('links')
movies = open_csv_data('movies')

# merge entre df_imdb et df_movie_lens
def merge_data():
    df_imdb = title_ratings.merge(right = title_crew, 
                                  how = 'inner',
                                  on = 'tconst').merge(right = title_basics,
                                                       how = 'inner',
                                                       on = 'tconst')
    
    df_movie_lens = ratings.merge(right = movies, how = 'inner', on = 'movieId').merge(right = links, how = 'inner', on = 'movieId')
    return df_imdb, df_movie_lens

df_imdb,df_movie_lens = merge_data()

def preprocessing_data():
    # suppression des colonnes tmdbId et timestamp
    df_movie_lens.drop({'tmdbId', 'timestamp'}, axis = 1, inplace = True)
    
    # remplacer les valeurs de tconst afin que ça soit fusionnable avec le df_movie_lens
    df_imdb['tconst'].replace({'tt':''}, regex= True,inplace = True)
    df_imdb['tconst'] = df_imdb['tconst'].astype('int')
    
    # renommer la colonne tconst en imdbId
    df_imdb.rename(columns= {'tconst' : 'imdbId'}, inplace = True)
    
    # merge df_movie_lens et df_imdb
    df_merged = df_movie_lens.merge(right = df_imdb, how = 'inner', on = 'imdbId')
    
    #suppression des colonnes inutiles, des données manquantes
    df_merged.drop(columns = {'endYear','title','originalTitle','genres_x','imdbId','isAdult'},inplace = True)
    df_merged.dropna(inplace = True)
    
    #modification de certaines colonnes pour les modèles et retourner les films ayant un minimum de 1000 votes (problème de mémoire sinon pour plus tard)
    df_merged['runtimeMinutes'] = df_merged['runtimeMinutes'].astype('int')
    df_merged['genres_y'].replace({',':' '}, regex = True, inplace = True)
    df_merged['writers'].replace({',':' '}, regex = True, inplace = True)
    df_merged['directors'].replace({',':' '}, regex = True, inplace = True)
    return df_merged[df_merged['numVotes']>=1000]

df_merged = preprocessing_data()

def separate_df():
    collab_filtering = df_merged.iloc[:,[0,1,2]]
    content_based_filtering = df_merged.iloc[:,[0,1,3,5,6,7,9,10,11]]
    return collab_filtering,content_based_filtering

collab_filtering,content_based_filtering = separate_df()


#############################################################################Content_based_filtering##################################################################################

def preprocessing_content_based_filtering():
    
    # supprimer les doublons
    content_based_filtering_duplicated = content_based_filtering.iloc[:,1:].drop_duplicates(keep = 'last')
    
    #standardiser les colonnes averageRating, startYear, runtimeMinutes
    scaler = MinMaxScaler()
    df_scaler = pd.DataFrame(scaler.fit_transform(content_based_filtering_duplicated.iloc[:,[1,5,6]]), 
                         columns= content_based_filtering_duplicated.iloc[:,[1,5,6]].columns)
    
    
    #create dict_movies
    dict_movie = pd.Series(df_merged['primaryTitle'].values,index = df_merged['movieId']).to_dict()
    content_based_filtering_duplicated.reset_index(inplace= True)
    movies = content_based_filtering_duplicated['movieId'].apply(lambda x : dict_movie[x])
    movies_index = pd.Series(movies.index, index = movies)
    
    # TfidVectorizer pour transformer les colonnes textuelles en vecteurs numériques se basant sur les mots et leur fréquence 
# d'apparition
    tfid = TfidfVectorizer(stop_words='english')
    tfid_genres = tfid.fit_transform(content_based_filtering_duplicated['genres_y'])
    tfid_directors = tfid.fit_transform(content_based_filtering_duplicated['directors'])
    tfid_writers = tfid.fit_transform(content_based_filtering_duplicated['writers'])
    
    # concaténer le df
    df_concat = pd.concat([df_scaler,
                       pd.DataFrame.sparse.from_spmatrix(tfid_genres),
                       pd.DataFrame.sparse.from_spmatrix(tfid_directors),
                       pd.DataFrame.sparse.from_spmatrix(tfid_writers)], axis = 1)
    
    df_concat = df_concat.astype('float16')
    return df_concat, movies,movies_index,content_based_filtering_duplicated

content_based_filtering, movies, movies_index,content_based_filtering_duplicated = preprocessing_content_based_filtering()

def train_nearest_neighbors_model(n_neighbors):
    model_knn = NearestNeighbors()
    model_knn.fit(content_based_filtering)
    return model_knn.kneighbors(np.array(content_based_filtering),n_neighbors+1,return_distance=True)

kneighbors_50 = train_nearest_neighbors_model(50)

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

recommendation_movies_knn('Toy Story')

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
    
    movies_sorted = collab_filtering_df.sort_values(['est'], ascending=False).merge(right = pd.DataFrame(content_based_filtering_duplicated[content_based_filtering_duplicated['movieId'].isin(collab_filtering_df['movieId'])]['movieId']).reset_index(),
                                                                                 on = 'movieId',
                                                                                 how = 'inner')
    movies_sorted = movies_sorted[~movies_sorted['movieId'].isin(collab_filtering[collab_filtering['userId'] == userId]['movieId'].unique())]
    return movies[movies_sorted['index']][0:n_recommendation]

def train_svd():
    reader = Reader(rating_scale=(0.5,5))
    data = Dataset.load_from_df(collab_filtering,reader = reader)
    
    trainset, testset = train_test_split(data, test_size=0.25,random_state=10)
    
    svd = SVD()
    svd.fit(trainset)
    
    test_pred = svd.test(testset)
    
    return svd,accuracy.rmse(test_pred, verbose=True) 

svd,rmse_svd = train_svd()

collaborative_filtering(userId = 309,
                        n_recommendation = 20,
                        svd_model = svd)


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
    ratings_from_content_filtering = collab_filtering[collab_filtering['movieId'].isin(collab_filtering[collab_filtering.index.isin(pd.Series(content_based_filtering_duplicated['index'],
                                                                                                         index = movies_index)[recommendation_movies_knn_user(movie,userId).index])]['movieId'].unique())]
    
   
    ratings_from_content_filtering['est'] = ratings_from_content_filtering.apply(lambda x: svd_model.predict(userId,x['movieId'], x['rating']).est, axis = 1)
    
    ratings_from_content_filtering = ratings_from_content_filtering.iloc[:,[1,3]].drop_duplicates(keep = 'first')
    
    movies_sorted = ratings_from_content_filtering.sort_values(['est'], ascending=False).merge(right = pd.DataFrame(content_based_filtering_duplicated[content_based_filtering_duplicated['movieId'].isin(ratings_from_content_filtering['movieId'])]['movieId']).reset_index(),
                                                                                 on = 'movieId',
                                                                                 how = 'inner')
    
    return movies[movies_sorted['index']][:20]

hybrid_recommendation_movies(userId = 108,
                                       movie = 'Toy Story',
                                       svd_model = svd)
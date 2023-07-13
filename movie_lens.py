import pandas as pd
import numpy as np
from datetime import date
import os
import pickle

#ouverture des différents fichiers
os.chdir('C:/Users/antho/Projet_recommandation_system')
ratings = pd.read_csv('ratings.csv')
ratings['timestamp'] = ratings['timestamp'].map(date.fromtimestamp)
links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')

# fusion des différents jeux de données (ratings - movies - links)
df_movie_lens = ratings.merge(right = movies, how = 'inner', on = 'movieId').merge(right = links, how = 'inner', on = 'movieId')

# suppression des colonnes tmdbId et timestamp
df_movie_lens.drop({'tmdbId', 'timestamp'}, axis = 1, inplace = True)

#ouverture des fichiers imdb
df_imdb = pd.read_pickle('df_imdb.pkl')
dict_person = pd.read_pickle('dict_person.pkl')

# remplacer les valeurs de tconst afin que ça soit fusionnable avec le df_movie_lens
df_imdb['tconst'].replace({'tt000000':'','tt00000':'','tt0000':'',
                                'tt000':'','tt00':'','tt0':'','tt':''}, regex= True, inplace = True)
df_imdb['tconst'] = df_imdb['tconst'].astype('int')

# renommer la colonne tconst en imdbId
df_imdb.rename(columns= {'tconst' : 'imdbId'}, inplace = True)

# merge df_movie_lens et df_imdb
df_merged = df_movie_lens.merge(right = df_imdb, how = 'inner', on = 'imdbId')

df_merged.drop(columns = {'endYear','title','originalTitle','genres_x','imdbId','nconst','category','isAdult'},inplace = True)
df_merged.dropna(inplace = True)
df_merged['runtimeMinutes'] = df_merged['runtimeMinutes'].astype('int')

df_merged['genres_y'].replace({',':' '}, regex = True, inplace = True)
df_merged['writers'].replace({',':' '}, regex = True, inplace = True)
df_merged['directors'].replace({',':' '}, regex = True, inplace = True)

# séparer les données en 2 dataframes et un dictionnaire
collab_filtering = df_merged.iloc[:,[0,1,2]]
content_based_filtering = df_merged.iloc[:,[0,1,3,5,6,7,9,10,11]]
dict_movie = pd.Series(df_merged['primaryTitle'].values,index = df_merged['movieId']).to_dict()

#sauvegarder les différentes variables
collab_filtering.to_pickle('collab_filtering.pkl')
content_based_filtering.to_pickle('content_based_filtering_bis.pkl')
with open('dict_person.pkl','wb') as dict_file:
    pickle.dump(dict_person,dict_file)
with open('dict_movie.pkl','wb') as dict_file:
    pickle.dump(dict_movie,dict_file)
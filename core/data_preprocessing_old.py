from datetime import date

import pandas as pd
import pickle
import os

os.chdir("D:\Github\MLE")

##### Téléchargement des différents inputs
## IMDB

def download_IMDB_data():
    title_ratings = pd.read_csv('data/title.ratings.tsv.gz',
                                compression='gzip', sep='\t', na_values='\\N')
    title_crew = pd.read_csv('data//title.crew.tsv.gz',
                             compression='gzip', sep='\t', na_values='\\N')
    name_basics = pd.read_csv('data/name.basics.tsv.gz',
                              compression='gzip', sep='\t', na_values='\\N')
    title_basics = pd.read_csv('data/title.basics.tsv.gz',
                               compression='gzip', sep='\t', na_values='\\N')
    title_principals = pd.read_csv('data/title.principals.tsv.gz',
                                   compression='gzip', sep='\t', na_values='\\N')
    return title_ratings, title_crew, name_basics, title_basics, title_principals


title_ratings, title_crew, name_basics, title_basics, title_principals = download_IMDB_data()


### Movie_lens

def download_Movie_lens_data():
    ratings = pd.read_csv('data/ratings.csv')
    links = pd.read_csv('data/links.csv')
    movies = pd.read_csv('data/movies.csv')
    return ratings, links, movies


ratings, links, movies = download_Movie_lens_data()

##### Fusion /Préprocessing

# IMDB data

# TODO : pk on réccupère pas les autres colonnes :  il manque = ordering, job et characters + est-ce qu'on fait bien de merge qu'en mode "inner"

def merged_imdb_data(df = title_principals):
    title_principals = df.groupby('tconst').agg(
    {'nconst': lambda x: ' '.join(x), 'category': lambda x: ' '.join(x)}).reset_index() # grouper les données de personId et category par film (5min)
    df_imdb = title_ratings.merge(right=title_crew,
                              how='inner',
                              on='tconst').merge(right=title_basics,
                                                 how='inner',
                                                 on='tconst').merge(right=title_principals,
                                                                    how='inner',
                                                                    on='tconst')

    # remplacer les valeurs de tconst afin que ça soit fusionnable avec le df_movie_lens
    df_imdb['tconst'].replace({'tt000000': '', 'tt00000': '', 'tt0000': '',
                               'tt000': '', 'tt00': '', 'tt0': '', 'tt': ''}, regex=True, inplace=True)
    df_imdb['tconst'] = df_imdb['tconst'].astype('int')

    # renommer la colonne tconst en imdbId
    df_imdb.rename(columns={'tconst': 'imdbId'}, inplace=True)

    return df_imdb

df_imdb = merged_imdb_data()

# Movie lens data
def merged_movie_lens_data(df = ratings):
    df_movie_lens = ratings.merge(right=movies, how='inner', on='movieId').merge(right=links, how='inner', on='movieId')
    df_movie_lens['timestamp'] = df_movie_lens['timestamp'].map(date.fromtimestamp)
    df_movie_lens.drop({'tmdbId', 'timestamp'}, axis=1, inplace=True)  # suppression des colonnes tmdbId et timestamp


# merge df_movie_lens et df_imdb
df_merged = df_movie_lens.merge(right=df_imdb, how='inner', on='imdbId')

df_merged.drop(columns={'endYear', 'title', 'originalTitle', 'genres_x', 'imdbId', 'nconst', 'category', 'isAdult'},
               inplace=True)
df_merged.dropna(inplace=True)
df_merged['runtimeMinutes'] = df_merged['runtimeMinutes'].astype('int')

df_merged['genres_y'].replace({',': ' '}, regex=True, inplace=True)
df_merged['writers'].replace({',': ' '}, regex=True, inplace=True)
df_merged['directors'].replace({',': ' '}, regex=True, inplace=True)

# séparer les données en 2 dataframes et un dictionnaire
collab_filtering = df_merged.iloc[:, [0, 1, 2]]
content_based_filtering = df_merged.iloc[:, [0, 1, 3, 5, 6, 7, 9, 10, 11]]
dict_movie = pd.Series(df_merged['primaryTitle'].values, index=df_merged['movieId']).to_dict()


# créer un dictionnaire personId : primaryName
dict_person = pd.Series(name_basics['primaryName'].values, index=name_basics['nconst']).to_dict()


# sauvegarder les différentes variables
collab_filtering.to_pickle('collab_filtering.pkl')
content_based_filtering.to_pickle('content_based_filtering_bis.pkl')
with open('dict_person.pkl', 'wb') as dict_file:
    pickle.dump(dict_person, dict_file)
with open('dict_movie.pkl', 'wb') as dict_file:
    pickle.dump(dict_movie, dict_file)

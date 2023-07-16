import os
import pandas as pd
import pickle
from movie_lens import df_merged

os.chdir("D:\Github\MLE")

#On décide d'avoir des inputs de test spécifiques différents lorsque l'on souhaite tester différentes conditions ou scénarios qui peuvent se produire dans le code.

def test_existing_files():
    assert os.path.exists('collab_filtering.pkl')
    assert os.path.exists('content_based_filtering_bis.pkl')
    assert os.path.exists('dict_person.pkl')
    assert os.path.exists('dict_movie.pkl')

def test_content_files():
    collab_filtering = pd.read_pickle('collab_filtering.pkl')
    content_based_filtering = pd.read_pickle('content_based_filtering_bis.pkl')
    dict_person = pickle.load(open('dict_person.pkl', 'rb'))
    dict_movie = pickle.load(open('dict_movie.pkl', 'rb'))
    assert isinstance(collab_filtering, pd.DataFrame)
    assert isinstance(content_based_filtering, pd.DataFrame)
    assert isinstance(dict_person, dict)
    assert isinstance(dict_movie, dict)
    assert len(collab_filtering) > 0
    assert len(content_based_filtering) > 0
    assert len(dict_person) > 0
    assert len(dict_movie) > 0

#
def test_data_cleaning():
    # Vérifiez que les colonnes sont correctement renommées
    #df_merged = pd.read_csv('data/merged_data.csv')  # Charger le DataFrame depuis le fichier CSV de données d'origine
    df_merged.drop(columns={'tmdbId', 'timestamp'}, axis=1, inplace=True)  # Appliquer la même suppression de colonnes
    assert set(df_merged.columns) == set(
        ['userId', 'movieId', 'rating', 'title', 'genres_y', 'startYear', 'runtimeMinutes', 'directors', 'writers',
         'primaryTitle', 'averageRating', 'numVotes'])

    # Vérifiez qu'il n'y a pas de valeurs manquantes
    assert not df_merged.isnull().any().any()

    # Vérifiez que les valeurs dans les colonnes genres_y, directors, et writers sont correctement remplacées
    assert not any(',' in genre for genre in df_merged['genres_y'])
    assert not any(',' in director for director in df_merged['directors'])
    assert not any(',' in writer for writer in df_merged['writers'])

    # Vérifiez que les colonnes inutiles sont supprimées
    assert 'endYear' not in df_merged.columns
    assert 'nconst' not in df_merged.columns
    assert 'category' not in df_merged.columns
    assert 'isAdult' not in df_merged.columns


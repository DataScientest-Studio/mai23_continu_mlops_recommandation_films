import os
import pandas as pd
import pickle

os.chdir("D:\Github\MLE")

# Remarque : chaque test ent ainsi indépendant

def test_existing_files():
    """
    On vérifie que les fichiers (base de données et dictionnaire) ont bien été crée et chargé pour les prochaines étapes
    """
    assert os.path.exists('df_imdb.pkl')
    assert os.path.exists('dict_person.pkl')

#
def test_content_files():
    """
    On vérifie le type des fichiers ainsi que leur taille
    """
    df_imdb = pd.read_pickle('df_imdb.pkl')
    name_basics = pd.read_csv('data/name.basics.tsv.gz',
                              compression='gzip', sep='\t', na_values='\\N')
    dict_person = pickle.load(open('dict_person.pkl', 'rb'))
    assert isinstance(df_imdb, pd.DataFrame)
    assert isinstance(dict_person, dict)
    assert len(df_imdb) > 0 # TODO : chercher à être plus précis, ex : le tconst similaire entre toutes les data entrées en input
    assert len(dict_person) == len(name_basics) # TODO : chercher à être plus précis

def test_dataframe_columns():
    """
    On vérifie la présence de certaines colonnes indispensables
    """
    df_imdb = pd.read_pickle('df_imdb.pkl')
    title_rating_col = ['tconst', 'averageRating', 'numVotes'] # TODO : quelles sont les colonnes essentielles
    title_crew = ['tconst', 'directors', 'writers']
    #name_basics = ['nconst','primaryName', 'birthYear', 'deathYear', 'primaryProfession', 'knownForTitles']
    title_basics = ['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'startYear', 'endYear', 'runtimeMinutes', 'genres']
    title_principals = ['tconst',  'nconst', 'category']
    expected_columns = title_rating_col + title_crew +  title_basics + title_principals
    expected_columns = list(set(expected_columns))
    assert all(col in df_imdb.columns for col in expected_columns)

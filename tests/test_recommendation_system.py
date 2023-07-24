import string
import pytest
from pandas import DataFrame
from surprise import accuracy, SVD

from recommendation_system import download_IMDB_data, download_Movie_lens_data
#drop_category, merge_category, merge_data, preprocessing_data, separate_df, preprocessing_content_based_filtering, train_nearest_neighbors_model
import os
import pandas as pd
os.chdir("D:\Github\MLE")


#title_ratings = pd.read_csv('data/sample/title_ratings.csv', sep=',')
#title_crew = pd.read_csv('data/sample/title_crew.csv', sep=',')
#title_basics = pd.read_csv('data/sample/title_basics.csv', sep=',')
#title_principals = pd.read_csv('data/sample/title_principals.csv', sep=',')

#ratings = pd.read_csv('data/sample/ratings_sample.csv')
#links = pd.read_csv('data/sample/links_sample.csv')
#movies = pd.read_csv('data/sample/movies_sample.csv')


# BDD
@pytest.fixture
def imdb_data():
    title_ratings, title_crew, title_basics, title_principals = download_IMDB_data(test=True)
    return title_ratings, title_crew, title_basics, title_principals


@pytest.fixture
def movielens_data():
    ratings, links, movies = download_Movie_lens_data(test=True)
    return ratings, links, movies



### TEST BDD

def test_download_IMDB_data(imdb_data):
    title_ratings, title_crew, title_basics, title_principals = imdb_data
    assert isinstance(title_ratings, DataFrame)
    assert isinstance(title_crew, DataFrame)
    #assert isinstance(name_basics, DataFrame)
    assert isinstance(title_basics, DataFrame)
    assert isinstance(title_principals, DataFrame)

def test_download_Movie_lens_data(movielens_data):
    ratings, links, movies = movielens_data
    assert isinstance(ratings, DataFrame)
    assert isinstance(links, DataFrame)
    assert isinstance(movies, DataFrame)



def test_merge_data():
    df_imdb, df_movie_lens = merge_data()
    assert isinstance(df_imdb, DataFrame)
    assert isinstance(df_movie_lens, DataFrame)

    imp_df_imdb = ["averageRating", "numVotes"] # TODO : to complete
    imp_df_movie_lens = ["movieId", "userId", "rating", "genres"]

    result_imdb = all(colonne in df_imdb.columns for colonne in imp_df_imdb)
    result_movie_lens = all(colonne in df_movie_lens.columns for colonne in imp_df_movie_lens)

    assert result_imdb is True
    assert result_movie_lens is True


def test_preprocessing_data():
    df_merged = preprocessing_data()
    assert "tconst" not in df_merged.columns
    assert not df_merged['ma_colonne'].str.startswith( 'tt').any()
    assert df_merged['titleType'].nunique() == 1
    assert df_merged['runtimeMinutes'].dtype == 'int'
    assert df_merged.isna().sum().sum() == 0
    col_to_delete = ['endYear','title','originalTitle','genres_x','imdbId','isAdult','actor','actress', 'directors', 'writers'] + ['tmdbId', 'timestamp']
    assert not set(col_to_delete).issubset(df_merged.columns)
    assert not df_merged["genres_y"].str.contains(f"[{string.punctuation}]").any()

def test_separate_df():
    collab_filtering, content_based_filtering = separate_df()
    col_collab_filtering = ["userId", "movieId", "ratings"]
    col_content_based_filtering = ['userId', "movieId", 'averageRating', 'titleType', 'startYear', 'runtimeMinutes', 'genres', 'director', 'writer', 'actors']
    assert col_collab_filtering.issubset(collab_filtering.columns)
    assert col_content_based_filtering.issubset(content_based_filtering.columns)


### MODELISATION TEST

def test_preprocessing_content_based_filtering():
    content_based_filtering, movies, movies_index, content_based_filtering_duplicated = preprocessing_content_based_filtering()
    assert not content_based_filtering_duplicated.duplicated().any()
    assert content_based_filtering.select_dtypes(exclude=['object']).max().max() == 1
    assert content_based_filtering.select_dtypes(exclude=['object']).min().min() == 0
    assert isinstance(movies, dict)
    assert content_based_filtering.duplicated().sum() == 0
    assert movies_index.iloc[15] == movies_index[15]


def test_train_nearest_neighbors_model(n_neighbors : int):
    n_neighbors = 3
    kneighbors = train_nearest_neighbors_model(n_neighbors = 3)
    assert isinstance(kneighbors, tuple)
    assert kneighbors[1].shape[1] == n_neighbors

def train_svd():
    svd, rmse = train_svd()
    assert 0<= rmse <1
    assert isinstance(svd, SVD)

def test_train_svd_with_high_rmse():
    collab_filtering_high_rmse = collab_filtering.copy()
    collab_filtering_high_rmse['rating'] = 2

    with pytest.raises(ValueError) as excinfo:
        train_svd(collab_filtering_high_rmse)

    assert "Le RMSE dépasse 1, il est recommandé de réentraîner le modèle." in str(excinfo.value)


def test_hybrid_recommandation_movies_bad_Ids():
    userId = 500000000000
    movie = 500000000000
    result = hybrid_recommandation_movies(userId, movie, n_recommandation = 3)
    assert str(result) == 'Erreur sur les paramètres rentrés dans la fonction, le userId et le film ne font pas partie de la base de données!'


def test_hybrid_recommandation_unknown_userID_only():
    userId = None
    movie = 5
    result = hybrid_recommandation_movies(userId, movie, n_recommandation = 3)
    assert len(result) == n_recommandation


def test_hybrid_recommandation_unknown_movieId_only():
    userId= 5
    movie = None
    result = hybrid_recommandation_movies(userId, movie, n_recommandation = 3)
    assert len(result) == n_recommandation

def test_hybrid_recommandation_all_Id_known():
    userId= 1
    movie = 'Jumanji'
    result = hybrid_recommandation_movies(userId, movie, n_recommandation = 3)
    assert isinstance(result, Series)



# Ajout test est bien du type prévu


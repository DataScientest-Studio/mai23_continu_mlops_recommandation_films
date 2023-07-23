import string
import pytest
from pandas import DataFrame

os.chdir("D:\Github\MLE")

title_ratings = pd.read_csv('data/title.ratings.tsv.gz',
                                compression='gzip', sep='\t', na_values='\\N')
title_rating
title_crew = pd.read_csv('data//title.crew.tsv.gz',
                             compression='gzip', sep='\t', na_values='\\N')
name_basics = pd.read_csv('data/name.basics.tsv.gz',
                              compression='gzip', sep='\t', na_values='\\N')
title_basics = pd.read_csv('data/title.basics.tsv.gz',
                               compression='gzip', sep='\t', na_values='\\N')
title_principals = pd.read_csv('data/title.principals.tsv.gz',
                                   compression='gzip', sep='\t', na_values='\\N')

ratings = pd.read_csv('data/ratings.csv')
links = pd.read_csv('data/links.csv')
movies = pd.read_csv('data/movies.csv')

"movieId" = [2,3,29,32]
"userId" = [1,5,9]

### CHECK que chaque fichier a bien été généré


def test_download_IMDB_data():
    assert isinstance(title_ratings, DataFrame)
    assert isinstance(title_crew, DataFrame)
    assert isinstance(name_basics, DataFrame)
    assert isinstance(title_basics, DataFrame)
    assert isinstance(title_principals, DataFrame)

def test_download_Movie_lens_data():
    ratings, links, movies = download_Movie_lens_data()
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
    assert not df_merged['ma_colonne'].str.startswith(
        'tt').any(), "Au moins une valeur commence par 'tt' dans la colonne 'ma_colonne'"
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

def test_preprocessing_content_based_filtering():
    content_based_filtering, movies, movies_index, content_based_filtering_duplicated = preprocessing_content_based_filtering()
    assert not content_based_filtering_duplicated.duplicated().any()
    assert content_based_filtering.select_dtypes(exclude=['object']).max().max() == 1
    assert content_based_filtering.select_dtypes(exclude=['object']).min().min() == 0






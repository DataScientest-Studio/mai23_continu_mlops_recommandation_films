import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split
#import mlflow

# define paths to data
data_path = "../data/"
data_path_tsv = "{}IMDB/".format(data_path)
data_path_csv = "{}Movie_Lens/ml-20m/".format(data_path)

# ouverture des fichiers
def open_IMDB_data(tsv_file, data_path_tsv = "../data/IMDB/"):
    return pd.read_csv('{path}{file}.tsv.gz'.format(path = data_path_tsv, file = tsv_file), compression='gzip',sep = '\t', na_values = '\\N', low_memory = False)

def open_Movie_lens_data(csv_file, data_path_csv = "../data/Movie_Lens/ml-20m/"):
    return pd.read_csv('{path}{file}.csv'.format(path = data_path_csv, file = csv_file))

title_ratings = open_IMDB_data(tsv_file='title.ratings')
title_crew = open_IMDB_data(tsv_file='title.crew')
title_basics = open_IMDB_data(tsv_file='title.basics')
title_principals = open_IMDB_data(tsv_file='title.principals')

ratings = open_Movie_lens_data('ratings')
links = open_Movie_lens_data('links')
movies = open_Movie_lens_data('movies')

# suppression des catégories de personnes avec beaucoup de NaNs
def drop_category():
    title_principals.drop(title_principals[title_principals['category'] == 'self'].index, inplace = True)
    title_principals.drop(title_principals[title_principals['category'] == 'cinematographer'].index, inplace = True)
    title_principals.drop(title_principals[title_principals['category'] == 'producer'].index, inplace = True)
    title_principals.drop(title_principals[title_principals['category'] == 'composer'].index, inplace = True)
    title_principals.drop(title_principals[title_principals['category'] == 'editor'].index, inplace = True)
    title_principals.drop(title_principals[title_principals['category'] == 'production_designer'].index, inplace = True)
    title_principals.drop(title_principals[title_principals['category'] == 'archive_footage'].index, inplace = True)
    title_principals.drop(title_principals[title_principals['category'] == 'archive_sound'].index, inplace = True)
    return title_principals

title_principals = drop_category()



    

# modifier title_principals pour avoir une colonne par catégories
def merge_category():
    # grouper les id des personnes par films et par leur rôle dans le film
    global title_principals
    title_principals = title_principals.groupby(['tconst','category']).agg({'nconst' : lambda x: ' '.join(x)}).reset_index()
    
    # créer des df par rôle
    for category in title_principals['category'].unique():
        globals()[category] = title_principals.groupby(by = ['category']).get_group(category).rename(columns={'nconst' : category}).drop('category', axis = 1)
    
    # merger les différents par rôle afin d'avoir un rôle par colonne
    title_principals_new = globals()[title_principals['category'].unique()[0]].merge(globals()[title_principals['category'].unique()[1]], how = 'outer', on = 'tconst')
    for i in range(2,len(title_principals['category'].unique())):
        title_principals_new = title_principals_new.merge(globals()[title_principals['category'].unique()[i]], how = 'outer', on = 'tconst')
    return title_principals_new



    

title_principals = merge_category()

# merge entre df_imdb et df_movie_lens
def merge_data():
    df_imdb = title_ratings.merge(right = title_crew, 
                                  how = 'inner',
                                  on = 'tconst').merge(right = title_basics,
                                                       how = 'inner',
                                                       on = 'tconst').merge(right = title_principals,
                                                                            how = 'inner',
                                                                            on = 'tconst')
    
    df_movie_lens = ratings.merge(right = movies, how = 'inner', on = 'movieId').merge(right = links, how = 'inner', on = 'movieId')
    return df_imdb, df_movie_lens

df_imdb,df_movie_lens = merge_data()


def preprocessing_data():
    # suppression des colonnes tmdbId et timestamp
    df_movie_lens.drop(['tmdbId', 'timestamp'], axis = 1, inplace = True)
    
    # remplacer les valeurs de tconst afin que ça soit fusionnable avec le df_movie_lens
    df_imdb['tconst'].replace({'tt':''}, regex= True,inplace = True)
    df_imdb['tconst'] = df_imdb['tconst'].astype('int')
    
    # renommer la colonne tconst en imdbId pour la fusion ci-après
    df_imdb.rename(columns= {'tconst' : 'imdbId'}, inplace = True)
    
    # merge df_movie_lens et df_imdb
    df_merged = df_movie_lens.merge(right = df_imdb, how = 'right', on = 'imdbId')

    # regrouper les colonnes actor et actress en une seule et remplacer les cases vides par des NaN
    df_merged['actors'] = df_merged['actor'].str.cat(df_merged['actress'],na_rep = '', sep=' ')
    df_merged.replace({' ' : np.nan}, inplace = True)
    
    #suppression des colonnes inutiles, des données manquantes
    df_merged.drop(columns = ['endYear','title','originalTitle','genres_x','isAdult','actor','actress', 'directors', 'writers'],inplace = True)
    #df_merged.dropna(inplace = True)
    
    
    #remplacement des ',' par des espaces afin d'utiliser ci-après la fonction tfid vectorizer
    df_merged['genres_y'].replace({',':' '}, regex = True, inplace = True)
    #retourner les films uniquement à partir des années 2000 (réduire la base de données)
    return df_merged[(df_merged['titleType']=='movie') & (df_merged['startYear']>=2000)]

df_merged = preprocessing_data()



def separate_df():
    collab_filtering = df_merged.iloc[:,[0,1,2]].dropna() #userId,movieId,ratings
    content_based_filtering = df_merged.iloc[:,[3,4,6,8,9,10,11,13]].dropna() #userId,movieId, averageRating, titleType, startYear,runtimeMinutes,genres, director, writer, actors
    # changement de type de la variable runtimeMinutes
    content_based_filtering['runtimeMinutes'] = content_based_filtering['runtimeMinutes'].astype('int')
    return collab_filtering,content_based_filtering

collab_filtering,content_based_filtering = separate_df()




#############################################################################models preprocessing and training#################################################################################

def preprocessing_content_based_filtering():
    
    # supprimer les doublons
    content_based_filtering_duplicated = content_based_filtering.drop_duplicates(keep = 'last')
    
    #standardiser les colonnes averageRating, startYear, runtimeMinutes (= integer)
    scaler = MinMaxScaler()
    df_scaler = pd.DataFrame(scaler.fit_transform(content_based_filtering_duplicated.iloc[:,[1,3,4]]), 
                             columns= content_based_filtering_duplicated.iloc[:,[1,3,4]].columns)
    
    
    #create dictionary pour les modèles et déterminer la liste des films
    dict_movie = pd.Series(df_merged['primaryTitle'].values,index = df_merged['imdbId']).to_dict() #dict --> movieId : movie {ex =  2 : Jumanji}
    content_based_filtering_duplicated.reset_index(inplace= True)
    movies = content_based_filtering_duplicated['imdbId'].apply(lambda x : dict_movie[x])  
    movies_index = pd.Series(movies.index, index = movies) #dict --> movie : movie_index (à partir de 0) {ex = Jumanji : 0}
    
    # TfidVectorizer pour transformer les colonnes textuelles en vecteurs numériques se basant sur les mots et leur fréquence 
# d'apparition
    tfid = TfidfVectorizer(stop_words='english')
    tfid_genres = tfid.fit_transform(content_based_filtering_duplicated['genres_y'])
    tfid_directors = tfid.fit_transform(content_based_filtering_duplicated['director'])
    #tfid_writers = tfid.fit_transform(content_based_filtering_duplicated['writer'])
    #tfid_actors = tfid.fit_transform(content_based_filtering_duplicated['actors'])

    # créer une liste des noms des colonnes pour avoir que des strings et non des entiers et des strings
    liste_genres = []
    #liste_actors = []
    liste_directors = []
    #liste_writers = []
    for i in range(tfid_genres.shape[1]):
        liste_genres.append('genres_{}'.format(i+1))
    #for i in range(tfid_actors.shape[1]):
        #liste_actors.append('actor_{}'.format(i+1))
    for i in range(tfid_directors.shape[1]):
        liste_directors.append('director_{}'.format(i+1))
    #for i in range(tfid_writers.shape[1]):
        #liste_writers.append('genres_{}'.format(i+1))

    
    # concaténer le df
    df_concat = pd.concat([df_scaler,
                       pd.DataFrame.sparse.from_spmatrix(tfid_genres, columns= liste_genres),
                       pd.DataFrame.sparse.from_spmatrix(tfid_directors, columns= liste_directors)], axis = 1)
    
    # réduire la taille du df float64 --> float16
    df_concat = df_concat.astype('float16')
    return df_concat, movies,movies_index,content_based_filtering_duplicated

content_based_filtering, movies, movies_index,content_based_filtering_duplicated = preprocessing_content_based_filtering()

collab_filtering.to_pickle('../data/loaded_api_datasets/collab_filtering_df.pkl')
content_based_filtering_duplicated.to_pickle('../data/loaded_api_datasets/content_based_filtering_df.pkl')
movies.to_pickle('../data/loaded_api_datasets/movies.pkl')
movies_index.to_pickle('../data/loaded_api_datasets/movies_index.pkl')
df_merged.to_pickle('../data/loaded_api_datasets/df_merged.pkl')

# entraînement des modèles

# nearestNeighbor - content based filtering
def train_nearest_neighbors_model(n_neighbors : int):
    # instancier le modèle
    model_knn = NearestNeighbors()

    #entraînement sur nos données
    model_knn.fit(content_based_filtering)

    # retourne une liste des indices des n plus proches voisins, le premier argument doit être au format array d'où le np.array, n_neighbors+1 car le premier film plus proche voisn est le film lui même
    return model_knn.kneighbors(np.array(content_based_filtering),n_neighbors+1,return_distance=True) 



kneighbors_50 = train_nearest_neighbors_model(50)

# svd model - collaborative filtering

def train_svd():
    #définir l'échelle des valeurs de notes
    reader = Reader(rating_scale=(0.5,5))
    #importer les données
    data = Dataset.load_from_df(collab_filtering,reader = reader)
    
    #séparer les données en un jeu de test et d'entraînement
    trainset, testset = train_test_split(data, test_size=0.25,random_state=10)
    
    # instancier le modèle SVD et l'appliquer sur les données d'entraînement
    svd = SVD()
    svd.fit(trainset)
    
    # tester sur nos données de test
    test_pred = svd.test(testset)
    
    #retourner le modèle et la rmse comme métrique
    return svd,accuracy.rmse(test_pred, verbose=True) 

svd,rmse_svd = train_svd()

mlflow.log_metric('rmse_svd',rmse_svd)

np.save('../data/loaded_api_datasets/kneighbors_results',kneighbors_50)
joblib.dump(svd,'../data/loaded_api_datasets/svd_model.pkl')


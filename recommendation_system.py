import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split

# define paths to data
data_path = "./data/"s
data_path_tsv = "{}IMDB/".format(data_path)
data_path_csv = "{}Movie_Lens/ml-20m/".format(data_path)

# ouverture des fichiers
def open_tsv_data(tsv_file, data_path_tsv = "./data/IMDB/"):
    return pd.read_csv('{path}{file}.tsv.gz'.format(path = data_path_tsv, file = tsv_file), compression='gzip',sep = '\t', na_values = '\\N')

def open_csv_data(csv_file, data_path_csv = "./data/Movie_Lens/ml-20m/"):
    return pd.read_csv('{path}{file}.csv'.format(path = data_path_csv, file = csv_file))

title_ratings = open_tsv_data(tsv_file='title.ratings')
title_crew = open_tsv_data(tsv_file='title.crew')
title_basics = open_tsv_data(tsv_file='title.basics')
title_principals = open_tsv_data(tsv_file='title.principals')

ratings = open_csv_data('ratings')
links = open_csv_data('links')
movies = open_csv_data('movies')

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
    
# modifier title_principals pour avoir une colonne par catégories
def merge_category():
    # grouper les id des personnes par films et par leur rôle dans le film
    title_principals = title_principals.groupby(['tconst','category']).agg({'nconst' : lambda x: ' '.join(x)}).reset_index()
    
    # créer des df par rôle
    for category in title_principals['category'].unique():
        globals()[category] = title_principals.groupby(by = ['category']).get_group(category).rename(columns={'nconst' : category}).drop('category', axis = 1)
    
    # merger les différents par rôle afin d'avoir un rôle par colonne
    title_principals = globals()[title_principals['category'].unique()[0]].merge(globals()[title_principals['category'].unique()[1]], how = 'outer', on = 'tconst')
    for i in range(2,len(title_principals['category'].unique())):
        title_principals = title_principals.merge(globals()[title_principals['category'].unique()[i]], how = 'outer', on = 'tconst')
    return title_principals
    
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
    df_merged.drop(columns = ['endYear','title','originalTitle','genres_x','imdbId','isAdult','actor','actress', 'directors', 'writers'],inplace = True)
    df_merged.dropna(inplace = True)
    
    # changement de type de la variable runtimeMinutes
    df_merged['runtimeMinutes'] = df_merged['runtimeMinutes'].astype('int')

    #remplacement des ',' par des espaces afin d'utiliser ci-après la fonction tfid vectorizer
    df_merged['genres_y'].replace({',':' '}, regex = True, inplace = True)
    #retourner les films uniquement
    return df_merged[df_merged['titleType']=='movie']

df_merged = preprocessing_data()

def separate_df():
    collab_filtering = df_merged.iloc[:,[0,1,2]] #userId,movieId,ratings
    content_based_filtering = df_merged.iloc[:,[0,1,3,5,7,8,9,10,11,12]] #userId,movieId, averageRating, titleType, startYear,runtimeMinutes,genres, director, writer, actors
    return collab_filtering,content_based_filtering

collab_filtering,content_based_filtering = separate_df()


#############################################################################hybrid_recommendation_system#################################################################################

def preprocessing_content_based_filtering():
    
    # supprimer les doublons
    content_based_filtering_duplicated = content_based_filtering.iloc[:,1:].drop_duplicates(keep = 'last')
    
    #standardiser les colonnes averageRating, startYear, runtimeMinutes (= integer)
    scaler = MinMaxScaler()
    df_scaler = pd.DataFrame(scaler.fit_transform(content_based_filtering_duplicated.iloc[:,[1,3,4]]), 
                             columns= content_based_filtering_duplicated.iloc[:,[1,3,4]].columns)
    
    
    #create dictionary pour les modèles et déterminer la liste des films
    dict_movie = pd.Series(df_merged['primaryTitle'].values,index = df_merged['movieId']).to_dict() #dict --> movieId : movie {ex =  2 : Jumanji}
    content_based_filtering_duplicated.reset_index(inplace= True)
    movies = content_based_filtering_duplicated['movieId'].apply(lambda x : dict_movie[x])  
    movies_index = pd.Series(movies.index, index = movies) #dict --> movie : movie_index (à partir de 0) {ex = Jumanji : 0}
    
    # TfidVectorizer pour transformer les colonnes textuelles en vecteurs numériques se basant sur les mots et leur fréquence 
# d'apparition
    tfid = TfidfVectorizer(stop_words='english')
    tfid_genres = tfid.fit_transform(content_based_filtering_duplicated['genres_y'])
    tfid_directors = tfid.fit_transform(content_based_filtering_duplicated['director'])
    tfid_writers = tfid.fit_transform(content_based_filtering_duplicated['writer'])
    tfid_actors = tfid.fit_transform(content_based_filtering_bis_duplicated['actors'])

    # créer une liste des noms des colonnes pour avoir que des strings et non des entiers et des strings
    liste_genres = []
    liste_actors = []
    liste_directors = []
    liste_writers = []
    for i in range(tfid_genres.shape[1]):
        liste_genres.append('genres_{}'.format(i+1))
    for i in range(tfid_actors.shape[1]):
        liste_actors.append('genres_{}'.format(i+1))
    for i in range(tfid_directors.shape[1]):
        liste_directors.append('genres_{}'.format(i+1))
    for i in range(tfid_writers.shape[1]):
        liste_writers.append('genres_{}'.format(i+1))

    
    # concaténer le df
    df_concat = pd.concat([df_scaler,
                       pd.DataFrame.sparse.from_spmatrix(tfid_genres, columns= liste_genres),
                       pd.DataFrame.sparse.from_spmatrix(tfid_actors, columns= liste_actors),
                       pd.DataFrame.sparse.from_spmatrix(tfid_directors, columns= liste_directors),
                       pd.DataFrame.sparse.from_spmatrix(tfid_writers, columns= liste_writers)], axis = 1)
    
    # réduire la taille du df float64 --> float16
    df_concat = df_concat.astype('float16')
    return df_concat, movies,movies_index,content_based_filtering_duplicated

content_based_filtering, movies, movies_index,content_based_filtering_duplicated = preprocessing_content_based_filtering()


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


#############################################################################Hybrid_recommandation_filtering##################################################################################

def hybrid_recommendation_movies(userId : int, movie : str, n_recommendation = 20, svd_model = svd):
    """
    userId : identifiant de l'utilisateur ;
    movie : nom du film ; 
    n_recommendation : nombre de films à recommander,
    svd_model : algorithme de SVD ;
    """
   
    if (userId not in df_merged['userId'].unique()) & (movie not in list(movies)):
        print('Erreur sur les paramètres rentrés dans la fonction, le userId et le film ne font pas partie de la base de données!')

    elif movie not in list(movies):
        """
        si le film n'est pas dans la liste, alors il retourne une liste de n films recommandés non déja vus par un userId se basant sur 
        les notes mises sur une liste de films et l'interaction avec d'autres utilisateurs et leurs notes (collaborative filtering)
        """


        global collab_filtering #pour instancier la variable collab_filtering qui est en dehors de la fonction
        collab_filtering_df = collab_filtering

        #supprimer les doublons, nous n'avons besoin que d'une donnée par film, l'entraînement a déjà été fait et gain de temps en calcul
        collab_filtering_df = collab_filtering_df.drop_duplicates(subset = ['movieId'],keep = 'first')

        #appliquer le modèle svd sur chaque ligne
        collab_filtering_df['est'] = collab_filtering_df.apply(lambda x: svd_model.predict(userId,x['movieId'], x['rating']).est,
                                                               axis = 1)
        
        #ne garder que la colonne movieId et la note estimée
        collab_filtering_df = collab_filtering_df.iloc[:,[1,3]]
        
        #trier le df par les notes estimées décroissante
        movies_sorted = collab_filtering_df.sort_values(['est'], ascending=False)

        
        movies_sorted = movies_sorted.merge(right = pd.DataFrame(content_based_filtering_duplicated[content_based_filtering_duplicated['movieId'].isin(collab_filtering_df['movieId'])]['movieId']).reset_index(),
                                                                                        on = 'movieId',
                                                                                        how = 'inner')
        
        # regarder quel film a déjà été vu par cet userId et ressortir les films non vus dans l'ordre des notes
        movies_sorted = movies_sorted[~movies_sorted['movieId'].isin(collab_filtering[collab_filtering['userId'] == userId]['movieId'].unique())]
        #retourne les n films
        return movies[movies_sorted['index']][0:n_recommendation]

    elif userId not in df_merged['userId'].unique():
        '''
        si l'utilisateur n'est pas dans la liste alors le système de recommandation se base uniquement sur le content_based_filtering et 
        les films les plus proches voisins en fonction des caractéristiques des films (nearestNeighbors méthode)

        explication ligne ci-dessous:
        movies_index[movie] --> retourne la ligne à laquelle se trouve le film en question dans le fichier movies
        kneighbors_50[1][movies_index[movie]] --> retourne la liste des indices des voisins du film en question du plus proche au plus loin
        au final, cela retourne les 20 films les plus proches voisins
        '''
        return movies.iloc[kneighbors_50[1][movies_index[movie]][1:n_recommendation+1]]
    
    else:
        # retourne la liste des films recommandés non vus par cet userId en fonction du content-based-filtering
        recommendation_movies_knn_user = movies.iloc[kneighbors_50[1][movies_index[movie]]][~movies.iloc[kneighbors_50[1][movies_index[movie]]].index.isin(collab_filtering[collab_filtering['userId'] == userId]['movieId'])][1:]
        
        ratings_from_content_filtering = collab_filtering[collab_filtering['movieId'].isin(collab_filtering[collab_filtering.index.isin(pd.Series(content_based_filtering_duplicated['index'],
                                                                                                                                                  index = movies_index)[recommendation_movies_knn_user.index])]['movieId'].unique())]
        
        #appliquer le modèle svd sur chaque ligne
        ratings_from_content_filtering['est'] = ratings_from_content_filtering.apply(lambda x: svd_model.predict(userId,x['movieId'], x['rating']).est, axis = 1)
        
        #supprimer les doublons, nous n'avons besoin que d'une donnée par film
        ratings_from_content_filtering = ratings_from_content_filtering.iloc[:,[1,3]].drop_duplicates(keep = 'first')
        
        #trier les films par notes décroissantes
        movies_sorted = ratings_from_content_filtering.sort_values(['est'], ascending=False)

        movies_sorted = movies_sorted.merge(right = pd.DataFrame(content_based_filtering_duplicated[content_based_filtering_duplicated['movieId'].isin(ratings_from_content_filtering['movieId'])]['movieId']).reset_index(),
                                                                                                   on = 'movieId',
                                                                                                   how = 'inner')
        
        return movies[movies_sorted['index']][:20]

hybrid_recommendation_movies(userId = 108,
                             movie = 'Toy Story',
                             svd_model = svd)



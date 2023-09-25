import os

import joblib
import pandas as pd
import numpy as  np
os.chdir('/Users/bayerobarry/PycharmProjects/mai23_continu_mlops_recommandation_films/')

kneighbors_results = np.load('data/loaded_api_datasets/kneighbors_results.npy')
svd_model = joblib.load('data/loaded_api_datasets/svd_model.pkl')
collab_filtering = pd.read_pickle('data/loaded_api_datasets/collab_filtering_df.pkl')
content_based_filtering_duplicated = pd.read_pickle('data/loaded_api_datasets/content_based_filtering_df.pkl')
movies = pd.read_pickle('data/loaded_api_datasets/movies.pkl')
movies_index = pd.read_pickle('data/loaded_api_datasets/movies_index.pkl')
df_merged = pd.read_pickle('data/loaded_api_datasets/df_merged.pkl')

def hybrid_recommendation_movies(userId: int, movie: str, n_recommendation=21, svd_model=svd_model,kneighbors_50 = kneighbors_results,
                                    ):
    """
    userId : identifiant de l'utilisateur ;
    movie : nom du film ; 
    n_recommendation : nombre de films à recommander,
    svd_model : algorithme de SVD ;
    """

    if (userId not in df_merged['userId'].unique()) & (movie not in list(movies)):
        raise ValueError(
                'Erreur sur les paramètres rentrés dans la fonction, le userId ou le film ne font pas partie de la base de données!')

    elif movie not in list(movies) :
        """
        si le film n'est pas dans la liste, alors il retourne une liste de n films recommandés non déja vus par un userId se basant sur 
        les notes mises sur une liste de films et l'interaction avec d'autres utilisateurs et leurs notes (collaborative filtering)
        """


        # supprimer les doublons, nous n'avons besoin que d'une donnée par film, l'entraînement a déjà été fait et gain de temps en calcul
        collab_filtering_df = collab_filtering.drop_duplicates(subset=['imdbId'], keep='first')

        # appliquer le modèle svd sur chaque ligne
        collab_filtering_df['est'] = collab_filtering_df.apply(
                lambda x: svd_model.predict(userId, x['imdbId'], x['rating']).est,
                axis=1)

        # ne garder que la colonne imdbId et la note estimée
        collab_filtering_df = collab_filtering_df.iloc[:, [1, 3]]

        # trier le df par les notes estimées décroissante
        movies_sorted = collab_filtering_df.sort_values(['est'], ascending=False)

        movies_sorted = movies_sorted.merge(right=pd.DataFrame(content_based_filtering_duplicated[
                                                                    content_based_filtering_duplicated['imdbId'].isin(
                                                                        collab_filtering_df['imdbId'])][
                                                                    'imdbId']).reset_index(),
                                                on='imdbId',
                                                how='inner')

        # regarder quel film a déjà été vu par cet userId et ressortir les films non vus dans l'ordre des notes
        movies_sorted = movies_sorted[
                ~movies_sorted['imdbId'].isin(collab_filtering[collab_filtering['userId'] == userId]['imdbId'].unique())]
        # retourne les n films
        return movies[movies_sorted['index']][0:n_recommendation]

    elif userId not in df_merged['userId'].unique() :
        '''
        si l'utilisateur n'est pas dans la liste alors le système de recommandation se base uniquement sur le content_based_filtering et 
        les films les plus proches voisins en fonction des caractéristiques des films (nearestNeighbors méthode)

        explication ligne ci-dessous:
        movies_index[movie] --> retourne la ligne à laquelle se trouve le film en question dans le fichier movies
        kneighbors_50[1][movies_index[movie]] --> retourne la liste des indices des voisins du film en question du plus proche au plus loin
        au final, cela retourne les 20 films les plus proches voisins
        '''
        liste = movies.iloc[kneighbors_50[1][movies_index[movie]][1:n_recommendation + 1]]
        mean_score = df_merged[(df_merged['primaryTitle'].isin(liste)) & df_merged['userId'].isin(df_merged[df_merged['primaryTitle'] == movie]['userId'].unique())]['averageRating'].mean()
    

        return liste, mean_score

            




    elif df_merged[df_merged['primaryTitle'] == movie]['rating'].any():

        # retourne la liste des films recommandés non vus par cet userId en fonction du content-based-filtering
        recommendation_movies_knn_user = movies.iloc[kneighbors_50[1][movies_index[movie]]][
                                                ~movies.iloc[kneighbors_50[1][movies_index[movie]]].index.isin(
                                                    collab_filtering[collab_filtering['userId'] == userId]['imdbId'])][1:]

        ratings_from_content_filtering = collab_filtering[collab_filtering['imdbId'].isin(
                collab_filtering[collab_filtering.index.isin(pd.Series(content_based_filtering_duplicated['index'],
                                                                    index=movies_index)[
                                                                recommendation_movies_knn_user.index])][
                    'imdbId'].unique())]

        # appliquer le modèle svd sur chaque ligne
        ratings_from_content_filtering['est'] = ratings_from_content_filtering.apply(
                lambda x: svd_model.predict(userId, x['imdbId'], x['rating']).est, axis=1)

        # supprimer les doublons, nous n'avons besoin que d'une donnée par film
        ratings_from_content_filtering = ratings_from_content_filtering.iloc[:, [1, 3]].drop_duplicates(keep='first')

        # trier les films par notes décroissantes
        movies_sorted = ratings_from_content_filtering.sort_values(['est'], ascending=False)

        movies_sorted = movies_sorted.merge(right=pd.DataFrame(content_based_filtering_duplicated[
                                                                    content_based_filtering_duplicated['imdbId'].isin(
                                                                        ratings_from_content_filtering['imdbId'])][
                                                                    'imdbId']).reset_index(),
                                                on='imdbId',
                                                how='inner')

        return movies[movies_sorted['index']][:n_recommendation]







    else:

        return movies.iloc[kneighbors_50[1][movies_index[movie]][1:n_recommendation + 1]]

        
            

import time
import pandas as pd
import numpy as np
import os
import gzip
import pickle

#ouverture des différents fichiers
title_ratings = pd.read_csv('C:/Users/antho/Projet_recommandation_system/title.ratings.tsv.gz', 
                            compression='gzip',sep = '\t', na_values = '\\N')
title_crew = pd.read_csv('C:/Users/antho/Projet_recommandation_system/title.crew.tsv.gz', 
                         compression='gzip', sep = '\t', na_values = '\\N')
name_basics = pd.read_csv('C:/Users/antho/Projet_recommandation_system/name.basics.tsv.gz', 
                          compression='gzip',sep = '\t', na_values = '\\N')
title_basics = pd.read_csv('C:/Users/antho/Projet_recommandation_system/title.basics.tsv.gz', 
                           compression='gzip',sep = '\t', na_values = '\\N')
title_principals = pd.read_csv('C:/Users/antho/Projet_recommandation_system/title.principals.tsv.gz', 
                               compression='gzip',sep = '\t', na_values = '\\N')

# grouper les données de personId et category par film (5min)
title_principals = title_principals.groupby('tconst').agg({'nconst' : lambda x: ' '.join(x),'category' : lambda x: ' '.join(x)}).reset_index()

# créer un dictionnaire personId : primaryName
dict_person = pd.Series(name_basics['primaryName'].values,index = name_basics['nconst']).to_dict()

# on merge les différents dataset entre eux
df_imdb = title_ratings.merge(right = title_crew, 
                              how = 'inner',
                              on = 'tconst').merge(right = title_basics,
                                                   how = 'inner',
                                                   on = 'tconst').merge(right = title_principals,
                                                                        how = 'inner',
                                                                        on = 'tconst')

# sauvegarde des fichiers
df_imdb.to_pickle('df_imdb.pkl')
with open('dict_person.pkl','wb') as dict_file:
    pickle.dump(dict_person,dict_file)
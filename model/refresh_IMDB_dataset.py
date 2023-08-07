import wget
import os

def refresh_IMDB_dataset(refresh = False):
    if refresh == True:

        # supprime les anciens fichiers IMDB
        os.remove('./data/IMDB/title.basics.tsv.gz')
        os.remove('./data/IMDB/title.crew.tsv.gz')
        os.remove('./data/IMDB/title.principals.tsv.gz')
        os.remove('./data/IMDB/title.ratings.tsv.gz')

        list_url = ['https://datasets.imdbws.com/title.principals.tsv.gz','https://datasets.imdbws.com/title.ratings.tsv.gz',
                    'https://datasets.imdbws.com/title.crew.tsv.gz','https://datasets.imdbws.com/title.basics.tsv.gz']
        
        #télécharger les fichiers IMDB les plus récents
        for url in list_url:
            wget.download(url, out = '../data/IMDB/')

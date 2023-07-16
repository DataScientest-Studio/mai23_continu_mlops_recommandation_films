# Système de recommandation de films


The README.md file is a central element of any git repository. It allows you to present your project, its objectives, and to explain how to install and launch the project, or even how to contribute to it.


Ce dépôt GitHub contient tout le code nécessaire pour le bon fonctionnement du projet que l'on a nommé **NOM DU PROJET**. Ce projet fil-rouge a été mené lors de notre formation [MLOPS](https://datascientest.com/formation-ml-ops) chez [DataScientest](https://datascientest.com/).

L'objectif est de mener à bien un projet de data science et d'élaborer un système de recommandation de films utilisable et destiné spécifiquement à notre future clientèle comme le ferait déjà certaines plateformes comme Netflix. Cela consiste donc en la création d'un modèle de Machine Learning prédisant une liste de films à partir de certaines informations rentrées en paramètres (userId,film, historique propre à chaque userId, etc), à sa mise en production et à son déploiement pour notre futur public. Nous allons donc utiliser certains outils comme une API, un système de conteneurisation comme Docker et des outils de déploiement et de monitoring (MLFlow, AirFlow, etc).

Ce projet a été mené par trois personnes en formation chez DataScientest:

- Fatoumata Barry [LinkedIn]() [GitHub](https://github.com/Barry-ta?tab=repositories)
- Anthony Ferre [LinkedIn](www.linkedin.com/in/anthony-ferre-6bb5b7172) [GitHub](https://github.com/anthoferre?tab=repositories)
- Thomas Head-Rapson [LinkedIn](https://www.linkedin.com/in/thomas-head-rapson-132008135/) [GitHub](https://github.com/thomasheadrapson?tab=repositories)

Pour faire fonctionner les différents fichiers présents dans ce dépôt, il faudra installer les différentes dépendances avec la commande suivante :

```
pip install -r requirements.txt
```

## Base de données
Les données qui ont été utilisées sont issues de deux sources web différentes :
- [MovieLens](https://grouplens.org/datasets/movielens/20m/) contient 20 millions de notes sur 27000 films différents. Plus de 138 000 personnes choisies aléatoirement ont alimentées cette base de données et ayant notées au moins 20 films chacun. Les données ont été récoltées entre 1995 et 2015. Cette base de données contient 6 fichiers différents : genome-score.csv / genome_tags.csv / links.csv / movies.csv / ratings.csv / tags.csv
- [IMDB](https://developer.imdb.com/non-commercial-datasets/) contient des caractéristiques de films(année de sortie, genre, réalisateur(s), écrivain(s),durée, note moyenne, etc). Cette base de données contient 7 fichiers différents : title.ratings / title.episode / title.crew / name.basics / title.akas / title.basics / title.principals

Une fois ces deux bases de données nettoyées et liées, nous avons un dataframe de plus de 19 millions de données donc très conséquents nous permettant de passer aux parties suivantes.

## API
comment faire fonctionner l'API

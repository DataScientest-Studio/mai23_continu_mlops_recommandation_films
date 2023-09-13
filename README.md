# Système de recommandation de films

Ce dépôt GitHub contient tout le code nécessaire pour le bon fonctionnement du projet que l'on a nommé **Movie-Recommendation**. Ce projet fil-rouge a été mené lors de notre formation [MLOPS](https://datascientest.com/formation-ml-ops) chez [DataScientest](https://datascientest.com/).

L'objectif est de mener à bien un projet de data science et d'élaborer un système de recommandation de films utilisable et destiné spécifiquement à notre future clientèle comme le ferait déjà certaines plateformes comme Netflix. Cela consiste donc en la création d'un modèle de Machine Learning prédisant une liste de films à partir de certaines informations rentrées en paramètres (userId,film, historique propre à chaque userId, etc), à sa mise en production et à son déploiement pour notre futur public. Nous allons donc utiliser certains outils comme une API, un système de conteneurisation comme Docker et des outils de déploiement et de monitoring (MLFlow, AirFlow, etc).

Ce projet a été mené par trois personnes en formation chez DataScientest:

- Fatoumata Barry [LinkedIn]() [GitHub](https://github.com/Barry-ta?tab=repositories)
- Anthony Ferre [LinkedIn](www.linkedin.com/in/anthony-ferre-6bb5b7172) [GitHub](https://github.com/anthoferre?tab=repositories)
- Thomas Head-Rapson [LinkedIn](https://www.linkedin.com/in/thomas-head-rapson-132008135/) [GitHub](https://github.com/thomasheadrapson?tab=repositories)

Pour faire fonctionner les différents fichiers présents dans ce dépôt, il faudra installer les différentes dépendances en se plaçant dans le dossier contenant le fichier requirements.txt avec la commande suivante :

```
pip install -r requirements.txt
```

## Base de données
Les données qui ont été utilisées sont issues de deux sources web différentes où elles peuvent être téléchargées et placées dans un dossier \data\Movie_Lens\ml-20m et \data\IMDB respectivement :
- [MovieLens](https://grouplens.org/datasets/movielens/20m/) contient 20 millions de notes sur 27000 films différents. Plus de 138 000 personnes choisies aléatoirement ont alimentées cette base de données et ayant notées au moins 20 films chacun. Les données ont été récoltées entre 1995 et 2015. Cette base de données contient 6 fichiers différents : genome-score.csv / genome_tags.csv / links.csv / movies.csv / ratings.csv / tags.csv
- [IMDB](https://developer.imdb.com/non-commercial-datasets/) contient des caractéristiques de films(année de sortie, genre, réalisateur(s), écrivain(s),durée, note moyenne, etc). Cette base de données contient 7 fichiers différents : title.ratings , title.episode , title.crew , name.basics , title.akas , title.basics et title.principals qui sont mis à jour quotidiennement. Afin de mettre à jour cette base de données IMDB chaque semaine, il est nécessaire de lancer le docker daemon et ensuite de se placer dans le répertoire \airflow et lancer la commande suivante :
```
docker-compose up
```

Une fois rendu sur l'url [**localhost:8080**](http://localhost:8080/home), il faut se rendre sur le DAG *my_dag_recommendation* et cliquer sur le bouton **PLAY** et les données sont mises à jour au bout de quelques minutes dans le dossier \data\IMDB .

Une fois ces deux bases de données nettoyées et liées, nous avons un dataframe de plus de 5 millions de données donc assez conséquents nous permettant de passer aux parties suivantes.

## Modèle

Afin de garantir une réponse rapide de l'API suite à une demande d'un utilisateur, il est nécessaire de sauvegarder nos différents modèles ainsi que certaines bases de données. Pour cela, il est nécessaire de lancer la commande suivante depuis son terminal en se placant dans le dossier \model :

```
python .\recommendation_system.py
```

Les différents fichiers sauvegardés au bout de plusieurs minutes vont se placer dans le dossier \data\loaded_api_datasets et serviront lors de la demande d'une requête via l'API.

## API
Afin de faire fonctionner l'API, il faudra lancer le docker daemon et ensuite se placer dans le répertoire \api contenant un fichier docker-compose.yml et lancer la commande depuis son terminal : 

```
docker-compose up
```

Une fois rendu sur l'url [**localhost:8000**](http://localhost:8000), plusieurs routes existent afin de répondre aux besoins des clients: un besoin lié aux informations des utilisateurs, un besoin lié aux notes mises sur les films et le système de recommendation en lui même.

Les routes *new_user* et *new_rating* permettent d'ajouter de nouvelles données à notre database en y insérant de nouveaux utilisateurs et de nouvelles notes permettant à notre système de recommandation de pouvoir évoluer dans le temps avec la sortie de nouveaux films et l'arrivée de nouveaux cinéphiles.

Les routes *delete_user* et *delete_rating* permettent quant à elles de supprimer des données erronées ou un utilisateur qui ne souhaite plus utiliser la plateforme.

Les routes *update_user* et *update_rating* permettent de modifier des données liées à un changement d'information sur l'utilisateur ou un ajustement sur une note d'un film.

Enfin la route *recommendation_system* renvoie une liste de 20 films les plus recommandés en se basant sur l'identifiant de l'utilisateur ou *userid* et le nom du film ou *movie*.

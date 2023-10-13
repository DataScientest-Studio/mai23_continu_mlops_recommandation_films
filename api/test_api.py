from pandas import DataFrame
import pytest
from fastapi.testclient import TestClient
from api import api, Rating
from typing import Optional
from pydantic import ValidationError

client = TestClient(api)


### Page d'accueil

def test_get_home():
    """
    Test que le lien fonctionne bien.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert 'Welcome to our API' in response.text


### Informations entrées par un utilisateur

# Fonctionne bien si bon format

def test_new_user_correct_information():
    """
    Test que sous le bon format on est un bien un message OK à l'issue
    """
    user_data = {

        "name": "Newuserr",
        "email": "newuser@gmail.com",
        "password": "userpssword"
    }

    response = client.post("/new_user", json=user_data)
    assert response.status_code == 200
    assert "userid" in response.text


# Fonctionne mal si mauvais format

def test_new_user_false_mail():
    """
    Test que si le mail d'entrée ne suit pas le bon format on est un message d'erreur (+ contrainte ajoutée dans l'API,
    car il faut indiquer à l'user dès la collecte des données)
    """
    invalid_user_data = {"name": "John Doe", "email": "falsemail.com", "password": "sshijh"}
    response = client.post("/new_user", json=invalid_user_data)
    assert response.status_code == 422
    assert response.json()['detail'] == "Invalid email format"





from fastapi.testclient import TestClient


class CustomTestClient(TestClient):
    def delete_with_payload(self, **kwargs):
        return self.request(method="DELETE", **kwargs)


def test_delete_user_existing():
    """
    Test que la suppression d'utilisateur fonctionne
    """
    client = CustomTestClient(api)
    existing_user_data = {"name": "Anthony", "email": "anthony@e.mail", "password": "abadpassword1"}
    response = client.delete_with_payload(url="/delete_user", json=existing_user_data)
    assert response.status_code == 200
    assert 'Success: True' in response.text


def test_update_user():
    """
    Test que l'actualisation d'information d'un utilisateur fonctionne
    """
    existing_user_data = {"userid": "1", "name": "Anthonynew", "email": "anthony@e.mail", "password": "abadpassword1"}
    response = client.patch("/update_user/", json=existing_user_data, params={"field": "name"})
    assert response.status_code == 200
    assert 'Success: True' in response.text



def test_update_fake_field():
    """
    Test que si un faux domaine est donné qu'on a bien un message d'erreur
    """
    existing_user_data = {
        "name": "Anthony",
        "email": "anthony@e.mail",
        "password": "abadpassword1"
    }

    response = client.patch("/update_user", json=existing_user_data, params={"field": "nonexistentfield"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid field: nonexistentfield"}


### Nouvelles notes entrées par l'utilsateur

import pytest
import sqlite3
import datetime
from test_db import test_db_schema

@pytest.fixture
def test_db():
    # Créez une base de données SQLite en mémoire pour les tests
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Exécutez le schéma de base de données de test
    cursor.executescript(test_db_schema)
    conn.commit()

    # Ajoutez des données de test si nécessaire
    # ...

    yield conn

    # Fermez la base de données après les tests
    conn.close()

#def test_update_rating(test_db):
    """
    Test que l'actualisation d'une note par un utilisateur
    """
#    client = TestClient(api)

    # Exécutez la mise à jour dans la base de données de test
#    with test_db:
#        cursor = test_db.cursor()
#        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("testuser", "testpassword"))
#        cursor.execute("INSERT INTO ratings (userid, movieid, rating) VALUES (?, ?, ?)", (1, 128734, 3))

#    rating_data = {"userid": 1, "movieid": 128734, "rating": 4}
#    response = client.patch("/update_rating", json=rating_data)
#    assert response.status_code == 200

#Fonctionne bien si bon format
#TODO : issue
def test_valid_rating():
    """
    Test que sous le bon format, l'attribution d'une nouvelle note fonctionne
    """
    client = TestClient(api)


    rating_data = {"userid": 1, "movieid": 128734, "rating": 5}
    response = client.post("/new_rating", json=rating_data)
    assert response.status_code == 200
    assert '{"ratingid":' in response.text



# Fonctionne mal si échelle non respectée

def test_invalid_low_rating(test_db):
    """
    Test que la note collectée respecte l'échelle minimale de notation
    """

    rating_data = {"userid": 0, "movieid": 129822, "rating": -1}
    response = client.post("/new_rating", json=rating_data)
    assert response.status_code != 200
    assert "greater than or equal to 0" in response.json()['detail'][0]['msg']



def test_invalid_high_rating(test_db):
    """
    Test que la note collectée respecte l'échelle maximale de notation
    """
    rating_data = {"userid": 0, "movieid": 128734, "rating": 10}
    response = client.post("/new_rating", json=rating_data)
    assert response.status_code != 200
    assert "less than or equal to 5" in response.json()['detail'][0]['msg']


def test_delete_ratings(test_db):
    """
    Test qu'une note peut bien être supprimée
    """

    client = CustomTestClient(api)

    existing_user_data = {"ratingid" : 0, "userid": 1, "movieid": 128734, "rating": 0}
    response = client.delete_with_payload(url="/delete_ratings", json=existing_user_data)
    assert response.status_code == 200
    assert response.json() == ['Success: True']


def test_update_rating():
    """'
    Test que l'actualisation d'une note pour un film  fonctionne
    """
    client = TestClient(api)

    rating_data = {"userid": 96, "movieid": 128734, "rating": 3}
    response = client.patch("/update_rating", json=rating_data)
    assert response.status_code == 200
    assert response.json() == ['Success: True']


def test_recommendation_system_valid():
    """
    Test que pour de bonnes données d'entrée le système de recommandation fonctionne
    """
    # Données pour la requête
    userid = 3453
    movie = "Oppenheimer"

    response = client.post(
        f"/recommendation_system?userid={userid}&movie={movie}",
        headers={"accept": "application/json"},
    )

    assert response.status_code == 200

    response_data = response.json()
    assert "When this route grows up it will provide recommendations for this movie: Oppenheimer" in response_data


import pytest



def test_recommendation_system_invalid_movie():
    """
    Test que pour de mauvaises données d'entrée le système de recommandation ne fonctionne pas
    """
    userid = 0
    movie = "Nonexistentmovie"

    with pytest.raises(ValueError) as e:
        response = client.post(
            f"/recommendation_system?userid={userid}&movie={movie}",
            headers={"accept": "application/json"},
        )

    assert 'Erreur sur les paramètres rentrés dans la fonction, le userId et le film ne font pas partie de la base de données!' in str(e.value)



def test_recommendation_system_invalid_user_ok():
    """
    Test que le pour un mauvais ID mais un bon film le système de recommandation ne fonctionne pas
    """
    userid = -5555
    movie = "Oppenheimer"

    response = client.post(
        f"/recommendation_system?userid={userid}&movie={movie}",
        headers={"accept": "application/json"},
    )

    assert response.status_code == 200
    assert 'recommendations for this movie:' in response.text


def test_log_event():
    """
    Test que le lien fonctionne bien.
    """
    event = {
        "userid": 123,
        "timestamp": 1678901234.235,
        "activity": "Login",
        "response_code": 200,
        "response_message": "Success",
        "output": {"info": "Additional info"}
    }
    response = client.post("/log_event", json=event)
    assert response.json() == ['This is the log_event route']
from pandas import DataFrame
import pytest
from fastapi.testclient import TestClient
from api import api
import init_db
from typing import Optional
from pydantic import ValidationError

client = TestClient(api)


### Page d'accueil

def test_get_home():
    """
    Test que le lien fonctionne bien.
    """
    response = client.get("/")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == ['Welcome to our API. This is a work in progress.']


### Informations entrées par un utilisateur

# Fonctionne bien si bon format

def test_new_user_correct_information():
    """
    Test que sous le bon format on est un bien un message OK à l'issue
    """
    user_data = {

        "name": "Newuserr",
        "email": "newuser@example.com",
        "password": "userpssword"
    }

    response = client.post("/new_user", json=user_data)
    assert response.status_code == 200


# Fonctionne mal si mauvais format

def test_new_user_false_mail():
    """
    Test que si le mail d'entrée ne suit pas le bon format on est un message d'erreur (+ contrainte ajoutée dans l'API,
    car il faut indiquer à l'user dès la collecte des données)
    """
    invalid_user_data = {"name": "John Doe", "email": "falsemail.com", "password": "sshijh"}
    response = client.post("/new_user", json=invalid_user_data)
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'value is not a valid email address: The email address is not valid. It must have exactly one @-sign.'


""" TODO: Remarque le code ne retourne pas de message d'erreur quand l'user existe déjà"""
def test_new_user_existing():
    """
    Test que si tentative d'ajout d'un user existant tout est ok
    """
    existing_user_data = {"name": "Anthony", "email": "anthony@e.mail", "password": "abadpassword1"}
    response = client.post("/new_user", json=existing_user_data)
    assert response.status_code == 200


from fastapi.testclient import TestClient

class CustomTestClient(TestClient):
    def delete_with_payload(self,  **kwargs):
        return self.request(method="DELETE", **kwargs)


def test_delete_user_existing():
    """
    Test que la suppression d'utilisateur fonctionne
    """
    client = CustomTestClient(api)
    existing_user_data = {"name": "Anthony", "email": "anthony@e.mail", "password": "abadpassword1"}
    response = client.delete_with_payload( url = "/delete_user", json=existing_user_data)
    assert response.status_code ==200



def test_update_user():
    existing_user_data = {"userid":"1","name": "Anthonynew", "email": "anthony@e.mail", "password": "abadpassword1"}
    response = client.patch("/update_user/", json=existing_user_data, params={"field": "name"})
    assert response.status_code == 200



def test_update_fake_field():
    existing_user_data = {
        "name": "Anthony",
        "email": "anthony@e.mail",
        "password": "abadpassword1"
    }
    
    response = client.patch("/update_user", json=existing_user_data, params={"field": "nonexistentfield"})
    
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid field: nonexistentfield"}


### Nouvelles notes entrées par l'utilsateur

# Fonctionne bien si bon format

def test_valid_rating():
    """
    Test que sous le bon format, l'attribution d'une nouvelle note fonctionne
    """
    rating_data = {"userid": 96, "movieid": 128734, "rating": 5}
    response = client.post("/new_rating", json=rating_data)
    assert response.status_code == 200


# Fonctionne mal si échelle non respectée

def test_invalid_low_rating():
    """
    Test que la note collectée respecte l'échelle minimale de notation
    """

    rating_data = {"userid": 0, "movieid": 129822, "rating": -1}
    response = client.post("/new_rating", json=rating_data)
    assert response.status_code != 200
    assert response.json()['detail'][0]['msg']== 'Input should be greater than or equal to 0' 



def test_invalid_high_rating():
    """
    Test que la note collectée respecte l'échelle maximale de notation
    """
    rating_data = {"userid": 0, "movieid": 128734, "rating": 10}
    response = client.post("/new_rating", json=rating_data)
    assert response.status_code != 200
    assert response.json()['detail'][0]['msg']== 'Input should be less than or equal to 5' 



def test_delete_ratings():
    """
    Test qu'une note peut bien être supprimée
    """

    client = CustomTestClient(api)

    existing_user_data = {"userid" : 1 ,"name": "Anthony", "email": "anthony@e.mail", "password": "abadpassword1"}
    response = client.delete_with_payload(url="/delete_ratings", json=existing_user_data)
    assert response.status_code == 200
    assert response.json() == ['Success: True']
    
    
def test_update_rating():
    """
    Test que l'actualisation d'une note par un utilisateur
    """
    rating_data = {"userid": 96, "movieid": 128734, "rating": 3}
    response = client.patch("/update_rating", json=rating_data)
    assert response.status_code == 200



def test_recommendation_system_valid():
    # Données pour la requête
    userId = 0
    movie = "Body"

    response = client.post(
        f"/recommendation_system?userId={userId}&movie={movie}",
        headers={"accept": "application/json"},
    )

    assert response.status_code == 200

    response_data = response.json()
    assert "When this route grows up it will provide recommendations for this movie: Body" in response_data


def test_recommendation_system_invalid_movie():
    userId = 0
    movie = "Nonexistentmovie"

    with pytest.raises(ValueError) as e:
        response = client.post(
            f"/recommendation_system?userId={userId}&movie={movie}",
            headers={"accept": "application/json"},
        )

    assert e.match("Erreur sur les paramètres rentrés dans la fonction, le userId ou le film ne font pas partie de la base de données!")

""" TODO : à modifier dans le code faut des cond"""
def test_recommendation_system_invalid_user_ok():
    userId = -5555
    movie = "Body"

    response = client.post(
        f"/recommendation_system?userId={userId}&movie={movie}",
        headers={"accept": "application/json"},
    )

    assert response.status_code == 200


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
    response = client.post("/log_event", json = event)
    assert response.json() == ['This is the log_event route']
from pandas import DataFrame
import pytest
from fastapi.testclient import TestClient
from api import api, Rating
from typing import Optional
from pydantic import ValidationError



client = TestClient(api)

# TODO: Version ultérieure qui intégrera le fait d'être en lien avec la bdd de base
#list_movieid = []
#list_userid = []

### Page d'accueil

def test_get_home():
    """
    Test que le lien fonctionne bien.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome to our API. This is a work in progress."}


### Informations entrées par un utilisateur

# Fonctionne bien si bon format

def test_new_user_correct_information():
    """
    Test que sous le bon format on est un bien un message OK à l'issue
    """
    user_data = {"name": "John Doe", "email": "new_user@gmail.com", "password": "sshijh"}
    response = client.put("/new_user", json=user_data)
    assert response.status_code == 200
    assert response.json() == {"This will add a new user."}


# Fonctionne mal si mauvais format

def test_new_user_false_mail():
    """
    Test que si le mail d'entrée ne suit pas le bon format on est un message d'erreur (+ contrainte ajoutée dans l'API,
    car il faut indiquer à l'user dès la collecte des données)
    """
    invalid_user_data = {"name": "John Doe", "email": "falsemail.com", "password": "sshijh"}
    response = client.put("/new_user", json=invalid_user_data)
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "email"],
                "msg": "value is not a valid email address",
                "type": "value_error.email",
            }
        ]
    } # ce qui devrait être renvoyé par l'API comme message d'erreur

def test_new_user_too_short_password():
    """
    Test que le mot de passe ait une taille minimale  (+ ajout de la contrainte dans le code pour l'API)
    """
    invalid_user_data = {"name": "John Doe", "email": "new_user@gmail.com", "password": "abs"}
    response = client.post("/new_user", json=invalid_user_data)
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "password"],
                "msg": "ensure this value has at least 6 characters",
                "type": "value_error.any_str.min_length",
                "ctx": {"limit_value": 6},
            }
        ]
    }


### Nouvelles notes entrées par l'utilsateur

# Fonctionne bien si bon format

def test_valid_rating():
    """
    Test que sous le bon format, l'attribution d'une nouvelle note fonctionne
    """
    rating_data = {"user_id": 123, "movie_id": 456, "rating": 4.5}
    response = client.post("/new_rating/", json=rating_data)
    assert response.status_code == 200
    assert response.json() == {"This will add a new rating"}


# Fonctionne mal si échelle non respectée

def test_invalid_low_rating():
    """
    Test que la note collectée respecte l'échelle minimale de notation
    """
    with pytest.raises(ValidationError) as exc_info:
        invalid_rating = Rating(score=-1)

def test_invalid_high_rating():
    """
    Test que la note collectée respecte l'échelle maximale de notation
    """
    with pytest.raises(ValidationError) as exc_info:
        invalid_rating = Rating(score=10)


### Système de recommandation

# Fonctionne bien si bon format
# TODO : manque check au niveau des inputs, qui doivent être dans un premier temps intégré dans le développement de l'API
def test_recommendation_system_code_OK():
    """
    Test que ça ne renvoie pas de message d'erreur quand les infos collectées sont sous le bon format
    """
    credentials_data = {"usenrid": "user123", "password": "password123"}
    movie_id = 12345
    response = client.post("/recommendation_system", json=credentials_data, json={"movieid": movie_id})
    assert response.status_code == 200



def test_recommendation_system_format():
    """
    Test que l'élément renvoyé suit un certain format
    """
    credentials_data = {"usenrid": "user123", "password": "password123"}
    movie_id = 12345
    response = client.post("/recommendation_system", json=credentials_data, json={"movieid": movie_id})
    data = response.json()
    try:
        df = DataFrame.from_dict(data)
    except ValueError:
        df = None

    assert isinstance(df, DataFrame)
    assert df.shape[0] == 20
    assert df.shape[1] == 1


### Log event


# Fonctionne bien si bon format

def test_log_event_correct_info():
    """
    Test que si tout éléments d'entrées ok qu'on ait pas de message d'erreur
    """
    event_data = {"userid": "some_event", "timestamp" : 54545, "activity" : " ?", "response_code" : 200,
                  "response_message" : "OK", "output" : "?"}
    response = client.post("/log_event", json=event_data)
    assert response.status_code == 200

# Fonctionne mal si mauvais format
def test_log_event_missing_info():
    """
    Test que si élément collecté manquant on ait un message d'erreur
    """
    event_data = {"userid": "some_event", "timestamp" : 54545, }
    response = client.post("/log_event", json=event_data)
    assert response.status_code == 422



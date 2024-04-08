import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


@pytest.fixture
def sample_input_data_greater_than_50k():
    return [
        {
            "age": 45,
            "workclass": "Private",
            "fnlwgt": 20000,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 15000,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
        }
    ]


@pytest.fixture
def sample_input_data_less_than_50k():
    return [
        {
            "age": 19,
            "workclass": "Private",
            "fnlwgt": 40000,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Handlers-cleaners",
            "relationship": "Own-child",
            "race": "Asian-Pac-Islander",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 35,
            "native_country": "United-States"
        }
    ]

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Predict if salary is greater or lower 50k based on census data"}

def test_predict_greater_than_50k(sample_input_data_greater_than_50k):
    response = client.post("/predict/", json=sample_input_data_greater_than_50k)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert predictions[0] == 1  # Assuming the model returns 1 for greater than 50k


def test_predict_less_than_50k(sample_input_data_less_than_50k):
    response = client.post("/predict/", json=sample_input_data_less_than_50k)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert predictions[0] == 0  # Assuming the model returns 0 for less than or equal to 50k

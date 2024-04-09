import requests
import json

data = [
    {
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 123456,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
]

json_data = json.dumps(data)

url = "https://nd0821-c3-starter-code-qz5o.onrender.com/predict/"

headers = {'Content-Type': 'application/json'}

response = requests.post(url, headers=headers, data=json_data)

if response.status_code == 200:
    print(response.json())
else:
    print("Request failed with status code:", response.status_code)
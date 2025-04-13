from fastapi.testclient import TestClient
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from app.server import app

client = TestClient(app)

#Positive case
def test_validUserID():
    user_id = 186719
    response = client.get(f"/recommend/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    reco = data["recommendations"]
    assert isinstance(reco, list)
    assert len(reco) <= 20



#Negative/Invalid cases
def test_negativeUserID():
    user_id = -1
    response = client.get(f"/recommend/{user_id}")
    
    assert response.status_code in [200,422]
    

def test_alphaUserID():
    user_id = 'abc'
    response = client.get(f"/recommend/{user_id}")
    
    assert response.status_code in [422]

def test_alphanumericUserID():
    user_id = 'ab@123'
    response = client.get(f"/recommend/{user_id}")
    
    assert response.status_code in [422]

def test_floatUserID():
    user_id = 3.14
    response = client.get(f"/recommend/{user_id}")
    
    assert response.status_code in [422]

#Null/Empty case
def test_nullUserID():
    user_id = ' '
    response = client.get(f"/recommend/{user_id}")
    
    assert response.status_code in [422]

#Edge cases
def test_smallestUserID():
    user_id = 0
    response = client.get(f"/recommend/{user_id}")
    
    assert response.status_code in [200,422]

def test_largestUserID():
    user_id = 9999999999
    response = client.get(f"/recommend/{user_id}")
    
    assert response.status_code in [200,422]

    
    
    
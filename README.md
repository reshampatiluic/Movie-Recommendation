## Running the server
1. ```docker compose build``` (skip to 2 if image already built)
2. ```docker compose up```

Prefix with sudo if there are permission issues

## Get Recommendations
```http://<ip-address>:8082/recommend/{user_id}```

## View docs
```http://<ip-address>:8082/docs``` or ```http://<ip-address>:8082/redoc```
# %%
import requests

# %%
url = 'http://localhost:9696/predict'

# %%
customer_id = 'xyz-123'

customer = {
 "seniority": 9,
 "home": 1,
 "time": 60,
 "age": 30,
 "marital": 2,
 "records": 1,
 "job": 3,
 "expenses": 73,
 "income": 129,
 "assets": 0,
 "debt": 0,
 "amount": 800,
 "price": 846}

# %%
response = requests.post(url, json=customer).json()
print(response)

# %%




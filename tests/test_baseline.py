import requests
import pandas as pd
import time

print("Loading one test row...")
# Grab the very first borrower from the test set
df = pd.read_parquet('../data/X_test.parquet').head(1)

# Convert the row into a flat JSON dictionary
payload = df.to_dict(orient='records')[0]

print("Sending payload to Microservice...")
start_time = time.time()

# Hit the local FastAPI endpoint
response = requests.post("http://127.0.0.1:8000/predict", json=payload)

network_latency = (time.time() - start_time) * 1000

print("\n" + "="*50)
print(f"Status Code: {response.status_code}")
print(f"API Response: {response.json()}")
print(f"Total Round-Trip Network Latency: {network_latency:.2f} ms")
print("="*50)
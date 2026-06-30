from locust import HttpUser, task, between
import random
import uuid

class FraudDetectionUser(HttpUser):
    wait_time = between(0.1, 1.0) # simulate realistic user wait time between requests

    @task
    def predict_fraud(self):
        # Generate some synthetic transaction data for load testing
        transaction_id = str(uuid.uuid4())
        user_id = f"user_{random.randint(1, 10000)}"
        amount = round(random.uniform(1.0, 5000.0), 2)
        merchant_id = f"merchant_{random.randint(1, 500)}"
        
        payload = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "amount": amount,
            "merchant_id": merchant_id,
            "timestamp": "2023-01-01T12:00:00Z"
        }

        with self.client.post("/v1/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")

    @task(3)
    def health_check(self):
        # Health check endpoint is hit more frequently by load balancers, but maybe less by users
        # Giving it a weight of 3 vs the default 1 for predict. Actually users don't hit health,
        # but it's good to test endpoint throughput.
        self.client.get("/health")

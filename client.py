import jwt
import requests
from datetime import datetime, timedelta

# Secret key for JWT
SECRET_KEY = "iloveyou"


# Generate a JWT token
def generate_jwt():
    payload = {
        "iss": "client-app",
        "exp": datetime.utcnow() + timedelta(minutes=10),  # Token expires in 10 minutes
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")



# Make authorized requests
def make_authorized_request(route, method="POST", data=None):
    token = generate_jwt()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"http://localhost:5000{route}"

    try:
        if method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "GET":
            response = requests.get(url, headers=headers)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}

        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Client functionalities
if __name__ == "__main__":
    # Load data from a CSV
    print("Loading data...")
    load_response = make_authorized_request(
        "/load-data",
        data={
            "file_path": "data.csv",
            "input_columns": ["square_feet", "bedrooms"],  # Correct column names
            "label_column": "price",  # Correct label column
        },
    )
    print(load_response)

    # Train the model
    print("\nTraining the model...")
    train_response = make_authorized_request("/train")
    print(train_response)

    # Save the model
    print("\nSaving the model...")
    save_response = make_authorized_request("/save")
    print(save_response)

    # Load the model
    print("\nLoading the model...")
    load_model_response = make_authorized_request("/load")
    print(load_model_response)

    # Make predictions
    print("\nMaking predictions...")
    predict_response = make_authorized_request(
        "/predict",
        data={"input_values": [[2000, 3]]},  # Example input: 2000 square feet, 3 bedrooms
    )
    print(predict_response)

    # Reset the model
    print("\nResetting the model...")
    reset_response = make_authorized_request("/reset")
    print(reset_response)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Path to save/load the model
MODEL_PATH = "model.pkl"


class ModelHandler:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.metrics = None

    def load_data(self, file_path, input_columns, label_column):
        """
        Load data from a CSV file.
        Args:
            file_path (str): Path to the CSV file.
            input_columns (list): Names of input columns.
            label_column (str): Name of the label column.
        Returns:
            Tuple (inputs, labels)
        """
        data = pd.read_csv(file_path)
        inputs = data[input_columns].values
        labels = data[label_column].values
        return inputs, labels

    def create_model(self):
        """Create a new Linear Regression model."""
        self.model = LinearRegression()

    def train_model(self, inputs, labels):
        """
        Train the model with the given inputs and labels using GridSearchCV.
        Args:
            inputs (array): Input features.
            labels (array): Target labels.
        Returns:
            dict: Training metrics (MSE, R2).
        """
        if self.model is None:
            self.create_model()

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {"fit_intercept": [True, False], "positive": [True, False]}  # Removed 'normalize'
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring="r2")
        grid_search.fit(X_train, y_train)

        # Update the model with the best estimator
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Evaluate the model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        self.metrics = {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "test_r2": test_r2,
        }
        return self.metrics

    def save_model(self):
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        joblib.dump(self.model, MODEL_PATH)

    def load_model(self):
        """Load the model from a file."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("No saved model found.")
        self.model = joblib.load(MODEL_PATH)

    def predict(self, input_values):
        """
        Make a prediction for given inputs.
        Args:
            input_values (array): Array of input values.
        Returns:
            array: Predictions.
        """
        if self.model is None:
            raise ValueError("No model loaded or trained.")
        return self.model.predict(np.array(input_values).reshape(-1, len(input_values[0]))).tolist()

    def reset_model(self):
        """Reset the model to an untrained state."""
        self.model = None
        self.best_params = None
        self.metrics = None

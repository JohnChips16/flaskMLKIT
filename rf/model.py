import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed to Classifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Updated metrics
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
        """Create a new Random Forest Classifier."""
        self.model = RandomForestClassifier(random_state=42)  # Initialize classifier

    def train_model(self, inputs, labels):
        """
        Train the model with the given inputs and labels using GridSearchCV.
        Args:
            inputs (array): Input features.
            labels (array): Target labels.
        Returns:
            dict: Training metrics (accuracy, classification report, confusion matrix).
        """
        if self.model is None:
            self.create_model()

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            inputs, labels, test_size=0.2, random_state=42
        )

        # Hyperparameter grid for Random Forest
        param_grid = {
            "n_estimators": [100, 200],  # Number of trees
            "max_depth": [None, 10, 20],  # Depth of trees
            "min_samples_split": [2, 5],  # Minimum samples to split a node
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring="accuracy",  # Use accuracy for classification
        )
        grid_search.fit(X_train, y_train)

        # Update the model with the best estimator
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Evaluate the model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred_test).tolist()  # Convert to list

        self.metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
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
            array: Predictions (class labels).
        """
        if self.model is None:
            raise ValueError("No model loaded or trained.")
        return self.model.predict(np.array(input_values).reshape(-1, len(input_values[0]))).tolist()

    def reset_model(self):
        """Reset the model to an untrained state."""
        self.model = None
        self.best_params = None
        self.metrics = None

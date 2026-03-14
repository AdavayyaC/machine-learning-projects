import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load Dataset
def load_data(path):
    df = pd.read_csv(path)
    return df


# 2. Preprocess Data
def preprocess_data(df):

    # Remove unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    return df


# 3. Split Features and Target
def split_data(df):

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y


# 4. Train Model
def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


# 5. Evaluate Model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


# 6. Save Model
def save_model(model, path):

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully!")


# 7. Main Pipeline
def main():

    # Load data
    df = load_data("./data/titanic.csv")

    # Preprocess data
    df = preprocess_data(df)

    # Feature & target split
    X, y = split_data(df)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, "./model/model.pkl")


# Run Script
if __name__ == "__main__":
    main()
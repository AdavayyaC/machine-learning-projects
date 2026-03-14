import pickle
import numpy as np

# -----------------------------
# 1. Load Saved Model
# -----------------------------
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# -----------------------------
# 2. Take Passenger Input
# -----------------------------
def get_passenger_data():

    print("Enter Passenger Details")

    pclass = int(input("Passenger Class (1/2/3): "))
    sex = int(input("Sex (0 = male, 1 = female): "))
    age = float(input("Age: "))
    sibsp = int(input("Number of siblings/spouses aboard: "))
    parch = int(input("Number of parents/children aboard: "))
    fare = float(input("Fare: "))
    embarked_q = int(input("Embarked Q (0 or 1): "))
    embarked_s = int(input("Embarked S (0 or 1): "))

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_q, embarked_s]])

    return features


# -----------------------------
# 3. Predict Survival
# -----------------------------
def predict(model, data):

    prediction = model.predict(data)

    if prediction[0] == 1:
        print("\nPrediction: Passenger Survived")
    else:
        print("\nPrediction: Passenger Did Not Survive")


# -----------------------------
# 4. Main
# -----------------------------
def main():

    model = load_model("./model/model.pkl")

    passenger_data = get_passenger_data()

    predict(model, passenger_data)


if __name__ == "__main__":
    main()
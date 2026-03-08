import pandas as pd
import pickle

# Load model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input as DataFrame with column names
new_house = pd.DataFrame([[2500, 4, 3, 5]], columns=['area','bedrooms','bathrooms','age'])

prediction = model.predict(new_house)
print(f"Predicted House Price: ₹{prediction[0]:,.0f}")
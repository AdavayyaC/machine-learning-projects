# House Price Prediction

A machine learning project that predicts house prices using regression models based on key property features.

## Overview

This project builds a predictive model to estimate house prices utilizing features such as living area, number of bedrooms and bathrooms, and property age. The model is trained on historical housing data and can make predictions on new properties.

## Features

The model uses the following features for price prediction:

- **Area (sqft)** - Total living area in square feet
- **Bedrooms** - Number of bedrooms
- **Bathrooms** - Number of bathrooms
- **Age** - Age of the house in years

## Algorithms Used

- **Linear Regression** - Primary algorithm for predicting house prices based on input features

## Project Structure

```
house_price_prediction/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/
│   └── house_data.csv       # Dataset with house features and prices
├── model/                   # Trained model files and artifacts
├── notebook/
│   └── analysis.ipynb       # Exploratory data analysis and visualization
└── src/
    ├── train.py             # Model training script
    └── predict.py           # Price prediction script
```

## Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd house_price_prediction
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On macOS/Linux
   ```

4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model on the dataset:

```bash
python src/train.py
```

This will:
- Load the training data from `data/house_data.csv`
- Prepare and preprocess the features
- Train the Linear Regression model
- Save the trained model to the `model/` directory

### Making Predictions

To predict house prices:

```bash
python src/predict.py
```

The script will:
- Load the trained model
- Accept input features (area, bedrooms, bathrooms, age)
- Output the predicted house price

### Exploratory Data Analysis

Open and run the Jupyter notebook for data exploration and visualization:

```bash
jupyter notebook notebook/analysis.ipynb
```

## Data

The dataset (`house_data.csv`) contains:
- House features: area, bedrooms, bathrooms, age
- Target variable: price (in dollars)

## Results

The trained model provides price predictions based on property characteristics. Refer to the analysis notebook for detailed performance metrics and visualizations.

## Requirements

See `requirements.txt` for all Python dependencies. Common packages include:
- numpy
- pandas
- scikit-learn
- jupyter
- matplotlib

## Contributing

Contributions are welcome! Feel free to:
- Report issues or bugs
- Suggest improvements
- Submit pull requests
 
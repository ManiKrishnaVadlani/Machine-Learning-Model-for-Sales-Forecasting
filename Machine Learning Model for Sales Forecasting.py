import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    
    # Drop unnecessary columns
    data.drop(['Date'], axis=1, inplace=True)
    
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    return data

def train_model(data):
    # Define features and target
    X = data.drop('Sales', axis=1)
    y = data['Sales']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")
    
    return model, X_test, y_test, y_pred

def visualize_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label="Actual Sales", marker='o')
    plt.plot(range(len(y_pred)), y_pred, label="Predicted Sales", marker='x')
    plt.title("Sales Forecasting: Actual vs Predicted")
    plt.xlabel("Data Points")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # File path to the dataset
    file_path = "sales_forecasting_data.csv"  # Replace with your dataset
    
    # Step 1: Load and preprocess data
    data = load_and_preprocess_data(file_path)
    
    # Step 2: Train the model
    model, X_test, y_test, y_pred = train_model(data)
    
    # Step 3: Visualize results
    visualize_results(y_test, y_pred)

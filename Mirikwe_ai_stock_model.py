# Ai Stock Prediction model
import subprocess
import sys

def install_missing_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ['tensorflow', 'numpy', 'pandas', 'matplotlib', 'scikit-learn']

# Install missing packages
install_missing_packages(required_packages)

# Ensure TensorFlow is installed and import it safely
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
except ImportError as e:
    print("Error: TensorFlow failed to install or import. Please check your installation.")
    sys.exit(1)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load Stock Data
def load_data(stock_file):
    df = pd.read_csv(stock_file, date_parser=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Preprocess Data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['Stock'].values.reshape(-1,1))
    return scaled_data, scaler

# Create Training and Testing Sets
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Build LSTM Model
def build_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train Model
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Predict and Visualize
def predict_and_plot(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    
    plt.figure(figsize=(14,5))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.legend()
    plt.show()

# Load and Prepare Data
file_path = input("Enter the path to your stock data CSV file: ")  # Prompt for file path #  Input CSV File
stock_data = load_data(file_path)
scaled_data, scaler = preprocess_data(stock_data)
X, y = create_sequences(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

# Split into Training and Testing Sets
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Train and Evaluate Model
model = build_model()
model = train_model(model, X_train, y_train)
predict_and_plot(model, X_test, y_test, scaler)
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download stock data for a given ticker
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Example: Download MSFT stock data
ticker = "MSFT"
start_date = "2015-01-01"
end_date = "2023-01-01"
data = download_stock_data(ticker, start_date, end_date)

# Display the first few rows of the data
print(data.head())

# Plot the closing price
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label=f'{ticker} Closing Price')
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


### Step 3: Preprocess the Data

#We will use the `Close` price for training and prediction. The data will be normalized and split into training and testing sets.


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Preprocess the data
def preprocess_data(data, feature='Close', lookback=50):
    # Extract the feature (e.g., 'Close' price)
    prices = data[feature].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # Create sequences of data for TCN
    X, y = [], []
    for i in range(lookback, len(prices_scaled)):
        X.append(prices_scaled[i-lookback:i, 0])
        y.append(prices_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape X for Conv1D (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# Preprocess the data
lookback = 50  # Number of past days to use for prediction
X_train, X_test, y_train, y_test, scaler = preprocess_data(data, lookback=lookback)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


### Step 4: Build the TCN Model

#We will use the `TCNBlock` and `build_tcn` functions from the earlier implementation.


import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, Activation, Add

# Define the TCN block
class TCNBlock(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.0):
        super(TCNBlock, self).__init__()
        self.conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')
        self.bn1 = BatchNormalization()
        self.activation1 = Activation('relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')
        self.bn2 = BatchNormalization()
        self.activation2 = Activation('relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.residual = Conv1D(filters, 1, padding='same')  # Residual connection

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)

        # Add residual connection
        residual = self.residual(inputs)
        return Add()([x, residual])

# Build the TCN model
def build_tcn(input_shape, num_blocks, filters, kernel_size, dilation_rates, dropout_rate=0.0):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for i in range(num_blocks):
        x = TCNBlock(filters, kernel_size, dilation_rates[i], dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1)(x)  # Output layer for regression
    model = tf.keras.Model(inputs, outputs)
    return model

# Define model parameters
input_shape = (lookback, 1)
num_blocks = 3
filters = 32
kernel_size = 3
dilation_rates = [1, 2, 4]  # Increasing dilation rates
dropout_rate = 0.2

# Build the model
model = build_tcn(input_shape, num_blocks, filters, kernel_size, dilation_rates, dropout_rate)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display the model summary
model.summary()

### Step 5: Train the Model

#We will train the TCN model on the training data and validate it on the testing data.


# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

### Step 6: Make Predictions

#We will use the trained model to make predictions on the test data and visualize the results.


# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

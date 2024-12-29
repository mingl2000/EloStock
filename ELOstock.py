import tensorflow as tf
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Elo functions
def calculate_expected_score(rating_a, rating_b):
    """Calculate the expected score of stock A against stock B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo_ratings(stocks, returns, k=32):
    """Update Elo ratings based on returns."""
    updated_ratings = stocks.copy()
    stock_list = list(returns.keys())
    
    for i in range(len(stock_list)):
        for j in range(i + 1, len(stock_list)):
            stock_a = stock_list[i]
            stock_b = stock_list[j]
            
            # Compare returns
            return_a = returns[stock_a]
            return_b = returns[stock_b]
            
            # Determine match outcome
            if return_a > return_b:
                actual_a, actual_b = 1, 0
            elif return_a < return_b:
                actual_a, actual_b = 0, 1
            else:
                actual_a, actual_b = 0.5, 0.5
            
            # Current ratings
            rating_a = stocks[stock_a]
            rating_b = stocks[stock_b]
            
            # Expected scores
            expected_a = calculate_expected_score(rating_a, rating_b)
            expected_b = calculate_expected_score(rating_b, rating_a)
            
            # Update ratings
            updated_ratings[stock_a] += k * (actual_a - expected_a)
            updated_ratings[stock_b] += k * (actual_b - expected_b)
    
    return updated_ratings

# Fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    daily_returns = data.pct_change().iloc[1:]  # Calculate daily returns
    return data, daily_returns

# Prepare dataset
def prepare_dataset(tickers, start_date, end_date):
    """Prepare features and target (Elo ratings) for machine learning."""
    data, returns = fetch_stock_data(tickers, start_date, end_date)
    stock_ratings = {ticker: 1500 for ticker in tickers}  # Initial Elo ratings
    
    # Calculate Elo ratings for each day
    elo_ratings = []
    dates = []
    for date, row in returns.iterrows():
        daily_returns = row.to_dict()
        stock_ratings = update_elo_ratings(stock_ratings, daily_returns)
        elo_ratings.append(list(stock_ratings.values()))
        dates.append(date)
    
    features = returns.values[1:]  # Daily returns as features
    target = np.array(elo_ratings[1:])  # Elo ratings as the target variable
    return features, target, dates

# TensorFlow model
def build_model(input_shape):
    """Build and compile a TensorFlow model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),  # Dropout layer to prevent overfitting
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(tickers))  # Output one value per stock
    ])
    
    # Experiment with a lower learning rate and a different optimizer (RMSprop)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

# Main script
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    start_date = "2024-01-01"
    end_date = "2024-12-27"
    
    # Prepare dataset
    features, target, dates = prepare_dataset(tickers, start_date, end_date)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)  # Scale features
    target = scaler.fit_transform(target)  # Scale targets

    # Split data without shuffling (so that the order is preserved)
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(features, target, dates[1:], test_size=0.2, shuffle=False)
    
    # Build and train the model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.1, verbose=1)

    # Test the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Predict Elo ratings for test data
    predictions = model.predict(X_test)
    print("\nPredicted vs. Actual Elo Ratings (Top 2 Tickers):")

    # Select the top 2 tickers based on predicted Elo ratings for each day
    for j in range(-5, 0):  # For each of the last 5 days in the test set
        predicted_elo = predictions[j]
        tickers_elo = list(zip(tickers, predicted_elo))  # Combine tickers with predicted Elo
        tickers_elo.sort(key=lambda x: x[1], reverse=True)  # Sort by Elo rating in descending order
        
        top_2_tickers = tickers_elo[:2]  # Get the top 2 tickers
        print(f"\nDate: {dates_test[j]}")
        print(f"Top 2 stocks to trade on {dates_test[j]}:")
        for ticker, elo in top_2_tickers:
            print(f"Ticker: {ticker}, Predicted Elo: {elo:.2f}")

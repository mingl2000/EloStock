import tensorflow as tf
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    return features, target, dates, data

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
    tickers = ["689009.SS","688981.SS","688819.SS","688777.SS","688728.SS",
               
               "688617.SS","688599.SS","688561.SS","688538.SS",
               "688349.SS","688303.SS","688301.SS","688297.SS","688295.SS",
               "688220.SS","688188.SS","688187.SS","688180.SS","688169.SS",
               "688126.SS","688122.SS","688120.SS","688114.SS","688111.SS",
               "688099.SS","688082.SS","688072.SS","688065.SS","688047.SS",
               "688041.SS","688036.SS","688012.SS","688009.SS","688008.SS",
               "688271.SS","688256.SS","688234.SS","688223.SS",
               "688396.SS","688385.SS","688375.SS",
               "688525.SS","688521.SS","688475.SS"]
    start_date = "2024-01-01"
    end_date = "2024-12-27"
    
    # Prepare dataset
    features, target, dates, data = prepare_dataset(tickers, start_date, end_date)
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
    
    # Simulate trading with $10,000 for top 2 stocks and equal split across all stocks
    initial_investment = 10000
    portfolio_value_top_2 = initial_investment
    portfolio_value_equal = initial_investment
    last_top_2 = None
    top_2_tickers = []

    portfolio_values_top_2 = [portfolio_value_top_2]  # Store portfolio value for top 2 stocks
    portfolio_values_equal = [portfolio_value_equal]  # Store portfolio value for equal split

    # Fetch daily returns for simulation from original data (not predictions)
    for j in range(len(dates_test) - 1):  # Loop until the second last date
        predicted_elo = predictions[j]
        tickers_elo = list(zip(tickers, predicted_elo))  # Combine tickers with predicted Elo
        tickers_elo.sort(key=lambda x: x[1], reverse=True)  # Sort by Elo rating in descending order
        
        top_2_today = tickers_elo[:2]  # Get the top 2 tickers
        
        # Update only if top 2 tickers change
        if top_2_today != last_top_2:
            last_top_2 = top_2_today
        
        stock_1 = top_2_today[0][0]
        stock_2 = top_2_today[1][0]
        
        # Get next day's returns for top 2 stocks
        try:
            next_day_return_1 = (data[stock_1].iloc[j + 1] / data[stock_1].iloc[j]) - 1
            next_day_return_2 = (data[stock_2].iloc[j + 1] / data[stock_2].iloc[j]) - 1
        except IndexError:
            continue  # Skip if next day's data is not available
        
        # Update portfolio based on next day's returns for top 2 stocks
        allocation_per_stock = portfolio_value_top_2 / 2  # Equal split between top 2 stocks
        portfolio_value_top_2 += allocation_per_stock * next_day_return_1 + allocation_per_stock * next_day_return_2
        
        # Simulate the portfolio with equal split across all stocks
        allocation_per_stock_equal = portfolio_value_equal / len(tickers)  # Equal split across all stocks
        for stock in tickers:
            try:
                next_day_return = (data[stock].iloc[j + 1] / data[stock].iloc[j]) - 1
                portfolio_value_equal += allocation_per_stock_equal * next_day_return
            except IndexError:
                continue  # Skip if next day's data is not available
        
        # Store the portfolio values for plotting
        portfolio_values_top_2.append(portfolio_value_top_2)
        portfolio_values_equal.append(portfolio_value_equal)
        
        # Print results for the day
        print(f"\nDate: {dates_test[j]}")
        print(f"Top 2 Stocks: {stock_1}, {stock_2}")
        print(f"Portfolio Value (Top 2 Stocks): ${portfolio_value_top_2:.2f}")
        print(f"Portfolio Value (Equal Split): ${portfolio_value_equal:.2f}")
    
    print(f"\nFinal Portfolio Value (Top 2 Stocks): ${portfolio_value_top_2:.2f}")
    print(f"Final Portfolio Value (Equal Split): ${portfolio_value_equal:.2f}")

    # Saving the plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test[:-1], portfolio_values_top_2[1:], label="Top 2 Stocks Portfolio", color='b', linewidth=2)
    plt.plot(dates_test[:-1], portfolio_values_equal[1:], label="Equal Split Portfolio", color='g', linewidth=2)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/mnt/d/PriProjects/EloStock/portfolio_simulation_updated.png')  # Save plot as a PNG file
    print("Plot saved as 'portfolio_simulation_updated.png'.")
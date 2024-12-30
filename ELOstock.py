import tensorflow as tf
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import os

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()
  plt.xscale('log')
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()
  plt.savefig('/mnt/d/PriProjects/EloStock/tensorflow_history.png')
  #plt.show()

def set_random_seed():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Elo functions
def calculate_expected_score(rating_a, rating_b):
    """Calculate the expected score of stock A against stock B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo_ratings(stocks, returns, k=32):
    """Update Elo ratings based on returns with magnitude weighting."""
    updated_ratings = stocks.copy()
    stock_list = list(returns.keys())
    
    for i in range(len(stock_list)):
        for j in range(i + 1, len(stock_list)):
            stock_a = stock_list[i]
            stock_b = stock_list[j]
            
            # Get returns for both stocks
            return_a = returns[stock_a]
            return_b = returns[stock_b]
            
            # Calculate actual outcomes with magnitude
            actual_a = max(0, min(1, (return_a - return_b) / (abs(return_a) + abs(return_b) + 1e-10)))
            actual_b = 1 - actual_a
            
            # Current ratings
            rating_a = stocks[stock_a]
            rating_b = stocks[stock_b]
            
            # Expected scores
            expected_a = calculate_expected_score(rating_a, rating_b)
            expected_b = calculate_expected_score(rating_b, rating_a)
            
            # Update ratings with magnitude-weighted outcomes
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

def training(checkpoint_path):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="max"  # 'max' for maximizing the monitored metric
    )
    # Build and train the model
    model = build_model(X_train.shape[1])
    history =model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.1, verbose=1, callbacks=[checkpoint_callback])
    plot_train_history(history,'multi Step Training and validation loss')
# Main script
if __name__ == "__main__":
    set_random_seed()
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
    tickers = "688981.ss,688599.ss,688561.ss,688396.ss,688363.ss,688303.ss,688187.ss,688169.ss,688126.ss,688111.ss,688065.ss,688036.ss,688012.ss,688008.ss,688005.ss,605499.ss,605117.ss,603993.ss,603986.ss,603899.ss,603882.ss,603833.ss,603806.ss,603799.ss,603659.ss,603501.ss,603486.ss,603392.ss,603369.ss,603290.ss,603288.ss,603260.ss,603259.ss,603195.ss,603185.ss,603019.ss,601998.ss,601995.ss,601989.ss,601988.ss,601985.ss,601966.ss,601939.ss,601919.ss,601901.ss,601899.ss,601898.ss,601888.ss,601881.ss,601878.ss,601877.ss,601868.ss,601865.ss,601857.ss,601838.ss,601818.ss,601816.ss,601808.ss,601800.ss,601799.ss,601788.ss,601766.ss,601728.ss,601698.ss,601689.ss,601688.ss,601669.ss,601668.ss,601658.ss,601633.ss,601628.ss,601618.ss,601615.ss,601601.ss,601600.ss,601398.ss,601390.ss,601377.ss,601360.ss,601336.ss,601328.ss,601319.ss,601318.ss,601288.ss,601238.ss,601236.ss,601229.ss,601225.ss,601216.ss,601211.ss,601186.ss,601169.ss,601166.ss,601155.ss,601138.ss,601117.ss,601111.ss,601100.ss,601088.ss,601066.ss,601021.ss,601012.ss,601009.ss,601006.ss,600999.ss,600989.ss,600958.ss,600941.ss,600926.ss,600919.ss,600918.ss,600905.ss,600900.ss,600893.ss,600887.ss,600886.ss,600884.ss,600845.ss,600837.ss,600809.ss,600803.ss,600795.ss,600763.ss,600760.ss,600745.ss,600741.ss,600690.ss,600674.ss,600660.ss,600606.ss,600600.ss,600588.ss,600585.ss,600584.ss,600570.ss,600547.ss,600519.ss,600460.ss,600438.ss,600436.ss,600426.ss,600406.ss,600383.ss,600362.ss,600346.ss,600332.ss,600309.ss,600276.ss,600233.ss,600219.ss,600196.ss,600188.ss,600183.ss,600176.ss,600150.ss,600132.ss,600115.ss,600111.ss,600104.ss,600089.ss,600085.ss,600061.ss,600050.ss,600048.ss,600039.ss,600036.ss,600031.ss,600030.ss,600029.ss,600028.ss,600025.ss,600019.ss,600018.ss,600016.ss,600015.ss,600011.ss,600010.ss,600009.ss,600000.ss,300999.sz,300979.sz,300957.sz,300919.sz,300896.sz,300782.sz,300769.sz,300763.sz,300760.sz,300759.sz,300751.sz,300750.sz,300661.sz,300628.sz,300601.sz,300595.sz,300529.sz,300498.sz,300496.sz,300454.sz,300450.sz,300433.sz,300413.sz,300408.sz,300347.sz,300316.sz,300274.sz,300223.sz,300207.sz,300142.sz,300124.sz,300122.sz,300059.sz,300033.sz,300015.sz,300014.sz,003816.sz,002938.sz,002920.sz,002916.sz,002841.sz,002821.sz,002812.sz,002756.sz,002736.sz,002714.sz,002709.sz,002648.sz,002602.sz,002601.sz,002600.sz,002594.sz,002555.sz,002493.sz,002475.sz,002466.sz,002460.sz,002459.sz,002415.sz,002414.sz,002410.sz,002371.sz,002352.sz,002311.sz,002304.sz,002271.sz,002252.sz,002241.sz,002236.sz,002230.sz,002202.sz,002180.sz,002179.sz,002142.sz,002129.sz,002120.sz,002074.sz,002064.sz,002050.sz,002049.sz,002032.sz,002027.sz,002008.sz,002007.sz,002001.sz,001979.sz,001289.sz,000977.sz,000963.sz,000938.sz,000895.sz,000877.sz,000876.sz,000858.sz,000800.sz,000792.sz,000786.sz,000776.sz,000768.sz,000733.sz,000725.sz,000723.sz,000708.sz,000661.sz,000651.sz,000625.sz,000596.sz,000568.sz,000538.sz,000425.sz,000408.sz,000338.sz,000333.sz"
    tickers =tickers.upper()
    tickers = tickers.split(",")
    #tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    start_date = "2024-01-01"
    end_date = "2024-12-27"
    
    # Prepare dataset
    features, target, dates, data = prepare_dataset(tickers, start_date, end_date)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)  # Scale features
    target = scaler.fit_transform(target)  # Scale targets

    # Split data without shuffling (so that the order is preserved)
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(features, target, dates[1:], test_size=0.4, shuffle=False)

    #checkpoint_path = "training/best.keras"
    checkpoint_path = "/mnt/d/PriProjects/EloStock/best_model.keras"
    training(checkpoint_path)
    best_model = tf.keras.models.load_model(checkpoint_path)

    # Test the model
    test_loss, test_mae = best_model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Predict Elo ratings for test data
    predictions = best_model.predict(X_test)
    
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
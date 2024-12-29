import yfinance as yf
import math

def calculate_expected_score(rating_a, rating_b):
    """Calculate the expected score of stock A against stock B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo_ratings(stocks, returns, k=32):
    """
    Update Elo ratings for a group of stocks based on their returns.
    
    Parameters:
        stocks: dict - Current ratings of stocks, e.g., {"AAPL": 1500, "MSFT": 1600}.
        returns: dict - Returns of stocks over a period, e.g., {"AAPL": 0.05, "MSFT": 0.03}.
        k: int - The K-factor to determine sensitivity of rating changes.
    
    Returns:
        updated_ratings: dict - Updated ratings of stocks.
    """
    updated_ratings = stocks.copy()
    
    # Get all stock pairs
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

def fetch_stock_returns(tickers, start_date, end_date):
    """
    Fetch stock returns for a given list of tickers and date range using Yahoo Finance.
    
    Parameters:
        tickers: list - List of stock tickers.
        start_date: str - Start date (YYYY-MM-DD).
        end_date: str - End date (YYYY-MM-DD).
    
    Returns:
        returns: dict - Returns of each stock as a percentage change.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    daily_returns = data.pct_change().iloc[-1]  # Get returns for the last available day
    return daily_returns.to_dict()

if __name__ == "__main__":
    # List of stock tickers
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    
    # Date range for returns
    start_date = "2024-12-20"
    end_date = "2024-12-27"
    
    # Fetch stock returns
    stock_returns = fetch_stock_returns(tickers, start_date, end_date)
    print("Stock Returns:", stock_returns)
    
    # Initial ratings for stocks
    stock_ratings = {ticker: 1500 for ticker in tickers}
    
    # Update ratings
    updated_ratings = update_elo_ratings(stock_ratings, stock_returns)
    
    # Print updated ratings
    print("\nUpdated Ratings:")
    for stock, rating in updated_ratings.items():
        print(f"{stock}: {round(rating, 2)}")

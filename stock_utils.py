"""
Utility functions for stock analysis and visualization.

This module contains reusable functions for:
- Fetching stock data
- Calculating moving averages
- Plotting stock charts with MA overlays
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# Hardcoded values added as named constants
LOOKBACK_YEARS = 3
MA_SHORT = 50
MA_LONG = 200
SYMBOL_EXCLUDE = ["-", "SOLV"]

def get_stock_data(symbol: str, 
                   start_date: datetime, 
                   end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch stock data for a given symbol using yfinance.
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL', '^DJI')
    start_date : datetime
        Start date for historical data
    end_date : datetime
        End date for historical data
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with columns: date, open, high, low, close, volume, adjusted
        Returns None if fetch fails
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            print(f"Warning: No data returned for {symbol}")
            return None
        
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Rename columns to match R style (lowercase)
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adjusted'
        }
        
        df = df.rename(columns=column_mapping)
        
        # If no 'Adj Close' exists (like for indices), use 'close' as 'adjusted'
        if 'adjusted' not in df.columns:
            df['adjusted'] = df['close']
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Reorder columns to match R output
        df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adjusted']]
        
        return df
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None


def calculate_moving_averages(df: pd.DataFrame, 
                              ma_short: int = MA_SHORT, 
                              ma_long: int = MA_LONG) -> pd.DataFrame:
    """
    Calculate moving averages for stock data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' column
    ma_short : int
        Short-term moving average period (default: 50)
    ma_long : int
        Long-term moving average period (default: 200)
    
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with ma_50 and ma_200 columns added
    """
    df = df.copy()
    
    # Calculate rolling means - match R's zoo::rollmean behavior
    # min_periods=window means NAs until we have enough data
    df['ma_50'] = df['close'].rolling(window=ma_short, min_periods=ma_short).mean()
    df['ma_200'] = df['close'].rolling(window=ma_long, min_periods=ma_long).mean()
    
    return df


def check_golden_cross(df: pd.DataFrame, 
                       as_of_date: Optional[datetime] = None) -> bool:
    """
    Check if MA50 >= MA200 (golden cross condition).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with ma_50 and ma_200 columns
    as_of_date : datetime, optional
        Date to check condition. If None, uses most recent date.
    
    Returns:
    --------
    bool
        True if golden cross condition is met, False otherwise
    """
    if as_of_date is None:
        latest = df.iloc[-1]
    else:
        latest = df[df['date'] == as_of_date].iloc[-1]
    
    return latest['ma_50'] >= latest['ma_200']


def filter_stocks_above_ma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter stocks where ma_50 >= ma_200 on the most recent date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with symbol, date, ma_50, ma_200 columns
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only stocks meeting criteria
    """
    # Get the most recent date for each symbol
    result = df.loc[df.groupby('symbol')['date'].idxmax()]
    
    # Filter where ma_50 >= ma_200
    result = result[result['ma_50'] >= result['ma_200']]
    
    return result[['symbol', 'date', 'close', 'ma_50', 'ma_200']]


def plot_stock_ma(df: pd.DataFrame, 
                  symbols_to_plot: List[str], 
                  title: str = "Stock Moving Average Analysis",
                  figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Plot stock prices with moving averages in faceted subplots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with symbol, date, close, ma_50, ma_200 columns
    symbols_to_plot : list
        List of stock symbols to plot
    title : str
        Overall plot title
    figsize : tuple
        Figure size (width, height)
    """
    import matplotlib.dates as mdates
    
    # Filter data for selected symbols
    df_plot = df[df['symbol'].isin(symbols_to_plot)].copy()
    
    if df_plot.empty:
        print("No data to plot")
        return
    
    # Ensure date column is datetime type
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    
    # Create date breaks like in R: start + (0:3)*365 + end
    min_date = df_plot['date'].min()
    max_date = df_plot['date'].max()
    
    # Generate yearly breaks
    date_breaks = [min_date + pd.Timedelta(days=i*365) for i in range(4)]
    date_breaks.append(max_date)
    
    # Determine grid layout
    n_stocks = len(symbols_to_plot)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_stocks > 1 else [axes]
    
    for idx, symbol in enumerate(sorted(symbols_to_plot)):
        ax = axes[idx]
        stock_data = df_plot[df_plot['symbol'] == symbol]
        
        # Plot lines
        ax.plot(stock_data['date'], stock_data['close'], label='Close', linewidth=1.5)
        ax.plot(stock_data['date'], stock_data['ma_50'], label='MA 50', linewidth=1.0)
        ax.plot(stock_data['date'], stock_data['ma_200'], label='MA 200', linewidth=1.0)
        
        # Set x-axis date formatting to match R
        ax.set_xticks(date_breaks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Formatting
        ax.set_title(symbol, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_stocks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# TEMP TEST VERSION OF FUNCTION WITH CSV READING INSTEAD OF HARDCODED SYMBOLS
def get_index_constituents(index_name: str = "DOW") -> pd.DataFrame:
    """
    Get constituents of a stock index.
    
    Currently reads from CSV file exported from R tidyquant::tq_index().
    """
    if index_name == "DOW":
        # Read CSV exported from R
        df = pd.read_csv("dow_constituents.csv", sep=";")
        return df
    else:
        raise NotImplementedError(f"Index {index_name} not implemented yet")

# COMMENTED OUT BELOW FOR NOW WHILE TESTING WITH A CSV 
# def get_index_constituents(index_name: str = "DOW") -> pd.DataFrame:
#     """
#     Get constituents of a stock index.
    
#     Note: This is a placeholder. In R you use tidyquant::tq_index().
#     For Python, you may need to maintain a manual list or use a data provider.
    
#     Parameters:
#     -----------
#     index_name : str
#         Index name (e.g., 'DOW', 'SP500')
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with columns: symbol, company, weight
#     """
#     # Placeholder - you'll need to populate this with actual index data
#     # Options: maintain CSV file, scrape from web, or use a data provider
    
#     if index_name == "DOW":
#         # Example: Dow Jones Industrial Average (as of 2026)
#         # You would populate this from a reliable source
#         symbols = [
#             "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", 
#             "CVX", "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", 
#             "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", 
#             "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
#         ]
        
#         return pd.DataFrame({
#             'symbol': symbols,
#             'company': symbols,  # Placeholder - add actual company names
#             'weight': [1.0/len(symbols)] * len(symbols)  # Equal weight placeholder
#         })
    
#     else:
#         raise NotImplementedError(f"Index {index_name} not implemented yet")


def print_analysis_summary(df_results: pd.DataFrame, 
                          total_symbols: int,
                          warning_count: int) -> None:
    """
    Print summary statistics of the analysis.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Filtered results DataFrame
    total_symbols : int
        Total number of symbols analyzed
    warning_count : int
        Number of symbols that generated warnings
    """
    n_passing = len(df_results)
    
    print("=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total symbols analyzed:        {total_symbols}")
    print(f"Symbols with data errors:      {warning_count}")
    print(f"Symbols passing MA criteria:   {n_passing}")
    print(f"Pass rate:                     {n_passing/total_symbols*100:.1f}%")
    print("=" * 60)
    print("\nSymbols above MA200:")
    print(df_results['symbol'].tolist())
    print("=" * 60)

# Stock Picker

A Python-based stock analysis tool for identifying potential buy candidates using moving average crossover strategies.

## Overview

This project evaluates stocks against technical selection criteria, specifically looking for "golden cross" patterns where the 50-day moving average (MA50) crosses above the 200-day moving average (MA200), indicating potential upward momentum.

## Features

- 📊 Stock screening based on moving average criteria
- 📈 Analysis with interactive plots
- 🔍 Supports major indices (DOW, S&P500)
- 📝 Historical tracking of market conditions and stock performance


## Project Structure

```
stock_picker/
├── README.md              # This file
├── stock_utils.py         # Reusable utility functions
├── analysis.ipynb         # Main analysis notebook
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock_picker.git
cd stock_picker
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Open the Jupyter notebook:
```bash
jupyter notebook analysis.ipynb
```

2. Run all cells to perform the analysis on the DOW index

3. Review the results:
   - Summary statistics showing how many stocks passed the criteria
   - Faceted plots showing price history with MA overlays
   - Filtered list of stocks meeting the death cross condition

### Configuration

Adjust parameters in `stock_utils.py`:

```python
LOOKBACK_YEARS = 3      # Years of historical data
MA_SHORT = 50           # Short-term moving average period
MA_LONG = 200           # Long-term moving average period
SYMBOL_EXCLUDE = [...]  # Symbols to exclude from analysis
```

## Analysis Approach

The tool implements a simple but effective strategy:

1. **Data Collection**: Fetch historical stock prices for index  
2. **Moving Averages**: Calculate MA50 and MA200 for each stock ticker
3. **Golden Cross Filter**: Identify stocks where MA50 is above MA200
4. **Visualization**: Create plots passing stocks with MA overlays for visual confirmation
5. **Tracking**: Document historical results and market context over time

## Dependencies

- `pandas` - Data manipulation
- `yfinance` - Stock data retrieval
- `matplotlib` - Plotting and visualization
- `seaborn` - Enhanced plotting aesthetics
- `jupyter` - Interactive notebook environment

See `requirements.txt` for library versions.

## Data Sources

The source of stock data is Yahoo Finance via the `yfinance` library. While free and convenient, be aware:
- Data is unofficial (scraped from Yahoo Finance)
- May have rate limits or occasional outages
- For production use, consider a paid data provider

## Contributing

This is a personal analysis tool, but suggestions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your own analysis strategies

## Disclaimer

**This tool is for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- The author is not responsible for any financial losses

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Original R implementation: `stock_picking_charts` repository
- Data provided by Yahoo Finance
- Inspired by classic technical analysis strategies

## Future Enhancements

- [ ] Add support for more indices (S&P 500, NASDAQ, etc.)
- [ ] Implement additional technical indicators (RSI, MACD)
- [ ] Backtest strategy performance
- [ ] Add email alerts for new golden crosses
- [ ] Portfolio tracking and position sizing

---

**Last Updated:** 2026-02-07
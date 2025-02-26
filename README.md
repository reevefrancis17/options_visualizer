# Options Chain Visualizer

A Python application for visualizing stock options data fetched from Yahoo Finance. This tool allows you to:
- View call and put options prices across different strike prices
- Toggle between different metrics (Spot Price, Last Price, Bid/Ask, Volume, etc.)
- Navigate through different expiration dates
- Track values at specific strike prices with interactive crosshairs
- Compare intrinsic and extrinsic option values

## Installation Steps

1. **Install Python**
   - Visit [python.org](https://python.org/downloads)
   - Download and install Python 3.11 or newer
   - During installation, make sure to check "Add Python to PATH"

2. **Download this Repository**
   - Click the green "Code" button above
   - Select "Download ZIP"
   - Extract the ZIP file to a folder on your computer
   - Or if you have git installed, run:
     ```
     git clone https://github.com/your-username/options-visualizer.git
     ```

3. **Install Required Packages**
   - Open a terminal/command prompt
   - Navigate to the extracted folder:
     ```
     cd path/to/options-visualizer
     ```
   - Install requirements:
     ```
     pip install -r requirements.txt
     ```

4. **Run the Application**
   - In the same terminal, run:
     ```
     cd python
     python offline_app.py
     ```
   - The application will start and automatically load SPY options data

## Using the Application

- Enter a stock symbol in the top-left text box (default is SPY) and click "Search"
- Use Previous/Next buttons to navigate between different expiration dates
- Select different metrics using the radio buttons at the top
- Move your mouse over the graph to see values at specific strike prices
- The current stock price is shown as a vertical green dashed line
- Calls are shown in blue, puts in red

## Data Source

This application fetches real-time options data from Yahoo Finance using the `yfinance` package. The data includes:
- Strike prices
- Bid/Ask prices
- Last price
- Volume and Open Interest
- Calculated values like intrinsic/extrinsic value

## Features

- **Progressive Loading**: Data loads incrementally, showing partial results while fetching continues
- **Complete Data Fetching**: Loads all available expiration dates without time limits
- **Threaded Data Fetching**: Uses background threads to keep the UI responsive during data loading
- **Automatic Data Fetching**: Automatically fetches and processes options data
- **Interactive Visualization**: Hover over the chart to see exact values
- **Multiple Metrics**: View different aspects of options data (price, volume, etc.)
- **Expiration Navigation**: Easily switch between different expiration dates
- **Performance Optimized**: Efficient rendering and throttled updates for smooth operation
- **Offline Mode**: Works entirely on your local machine

## Troubleshooting

If you encounter any issues:
1. Make sure you have an internet connection (required for data fetching)
2. Verify Python is installed correctly: `python --version`
3. Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`
4. Check the error logs in the `debug` directory
5. For rate limit errors, try again later as Yahoo Finance may have rate limits

## Notes

- Data is fetched in real-time from Yahoo Finance
- Market data is only available during market hours
- Some stocks might have limited options data available
- The application will load all available expiration dates for a ticker
- The progressive loading feature allows you to see and interact with partial data while more is being fetched
- For tickers with many expiration dates, the app will continue loading in the background while allowing you to interact with the data that's already loaded

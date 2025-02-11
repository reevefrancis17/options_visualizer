# Options Chain Visualizer

A Python application for visualizing stock options data fetched from Yahoo Finance. This tool allows you to:
- View call and put options prices across different strike prices
- Toggle between different metrics (Mid Price, Bid/Ask, Volume, etc.)
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
     pip install -r python/requirements.txt
     ```

4. **Run the Application**
   - In the same terminal, run:
     ```
     python offline_app.py
     ```

## Using the Application

- Enter a stock symbol in the top-right text box (default is AAPL)
- Use Previous/Next buttons to change expiration dates
- Select different metrics from the radio buttons on the left
- Toggle calls/puts using the checkboxes
- Move your mouse over the graph to see values at specific strikes
- The legend updates automatically to show values at your cursor position

## Data Source

This application fetches real-time options data from Yahoo Finance using the `yfinance` package. The data includes:
- Strike prices
- Bid/Ask prices
- Volume and Open Interest
- Calculated values like intrinsic/extrinsic value

## Troubleshooting

If you encounter any issues:
1. Make sure you have an internet connection (required for data fetching)
2. Verify Python is installed correctly: `python --version`
3. Try reinstalling requirements: `pip install -r python/requirements.txt --force-reinstall`
4. For invalid symbols, the app will revert to AAPL

## Notes

- Data is fetched in real-time from Yahoo Finance
- Market data is only available during market hours
- Some stocks might have limited options data available

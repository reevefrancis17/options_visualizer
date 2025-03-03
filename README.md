# Options Chain Visualizer

A web application for visualizing stock options data fetched from Yahoo Finance. This tool allows you to:
- View call and put options prices across different strike prices
- Toggle between different metrics (Price, Bid/Ask, Volume, etc.)
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

4. **Run the Web Application**
   - In the same terminal, run:
     ```
     python main.py
     ```
   - The web application will start and be available at http://localhost:5000 (or the port specified by the PORT environment variable)

## Project Structure

The project has been consolidated into a single server application:

- `main.py` - Entry point that runs the web application
- `options_visualizer_web/` - Contains the web application code
  - `app.py` - Flask application with routes and API endpoints
  - `static/` - JavaScript, CSS, and other static assets
  - `templates/` - HTML templates
- `options_visualizer_backend/` - Contains backend utilities and data management
  - `add_ticker.py` - Utility to add tickers to the watchlist
  - `clean_cache.py` - Utility to clean the data cache

## Using the Web Application

- Enter a stock symbol in the search box and click "Search"
- The application will fetch and display options data for the symbol
- Use the expiration date selector to navigate between different expiration dates
- The current stock price is shown as a vertical green dashed line
- Calls are shown in blue, puts in red
- Data loads progressively, allowing you to see partial results while fetching continues

## Data Source

This application fetches real-time options data from Yahoo Finance using the `yfinance` package. The data includes:
- Strike prices
- Bid/Ask prices
- Last price
- Volume and Open Interest
- Calculated values like intrinsic/extrinsic value

## Features

- **Progressive Loading**: Data loads incrementally, showing partial results while fetching continues
- **Background Data Fetching**: Data is fetched asynchronously to keep the UI responsive
- **Caching**: Data is cached to improve performance and reduce API calls
- **Interactive Visualization**: Hover over the chart to see exact values
- **Multiple Metrics**: View different aspects of options data (price, volume, etc.)
- **Expiration Navigation**: Easily switch between different expiration dates
- **Performance Optimized**: Efficient rendering and throttled updates for smooth operation
- **Black-Scholes Model**: Includes calculations for option pricing and Greeks

## Data Caching

The application uses a robust SQLite-based caching system to store options data locally:

- **Persistent Storage**: Options data is cached in a SQLite database for improved performance and reduced API calls
- **Automatic Refresh**: Cache entries older than 10 minutes are automatically refreshed in the background
- **Non-Blocking**: Stale cache entries are still returned while a refresh happens in the background
- **Shared Cache**: Both frontend and backend use the same cache to minimize API calls
- **Background Polling**: The entire cache is refreshed every 10 minutes to keep data current
- **Persistent Size**: The cache never shrinks, ensuring historical data remains available
- **Progressive Loading**: Partial data is cached immediately, allowing the UI to update as data is fetched

### Cache Management

You can manage the cache using the backend API:

```bash
# Get cache statistics
curl http://localhost:5002/api/cache/stats

# View available tickers in the cache
curl http://localhost:5002/api/tickers

# Add a ticker to the watchlist (will be cached)
curl -X POST -H "Content-Type: application/json" -d '{"ticker":"AAPL"}' http://localhost:5002/api/tickers/add

# Remove a ticker from the watchlist
curl -X POST -H "Content-Type: application/json" -d '{"ticker":"AAPL"}' http://localhost:5002/api/tickers/remove
```

The cache is stored in a platform-specific location:
- **Windows**: `%APPDATA%\options_visualizer\options_cache.db`
- **macOS/Linux**: `~/.cache/options_visualizer\options_cache.db`

## Troubleshooting

If you encounter any issues:
1. Make sure you have an internet connection (required for data fetching)
2. Verify Python is installed correctly: `python --version`
3. Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`
4. Check the error logs in the `logs` directory
5. For rate limit errors, try again later as Yahoo Finance may have rate limits

## Health Check

The application includes a health check endpoint at `/health` that returns the current status of the application.

## Notes

- Data is fetched in real-time from Yahoo Finance
- Market data is only available during market hours
- Some stocks might have limited options data available
- The application will load all available expiration dates for a ticker
- The progressive loading feature allows you to see and interact with partial data while more is being fetched

## Running the Application

### Standard Run

To run the application with both frontend and backend on separate threads:

```bash
python main.py
```

This will:
- Start the backend API server on port 5002 (or the port specified in the BACKEND_PORT environment variable)
- Start the frontend web server on port 5001 (or the port specified in the PORT environment variable)

### Environment Variables

You can customize the application using these environment variables:

- `PORT`: Frontend web server port (default: 5001)
- `BACKEND_PORT`: Backend API server port (default: 5002)
- `BACKEND_URL`: Backend API URL (default: http://localhost:5002)

Example:
```bash
PORT=8080 BACKEND_PORT=8081 python main.py
```

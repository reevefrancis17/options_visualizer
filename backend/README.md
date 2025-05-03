# Options Visualizer Backend

A Flask-based backend server for the Options Visualizer application. This server provides a stable, always-running service that fetches, caches, and serves options data from Yahoo Finance.

## Features

- Fetches options data from Yahoo Finance using the `yfinance` library
- Maintains a local cache of options data to reduce API calls
- Automatically refreshes cached data every 10 minutes
- Provides RESTful API endpoints for the frontend to consume
- Logs all server activities for monitoring and debugging

## Directory Structure

```
options_visualizer_backend/
├── app.py                  # Main Flask application
├── cache/                  # Directory for storing cached data files
│   └── [ticker].pkl        # Cached options data files (e.g., AAPL.pkl)
├── data/
│   └── tickers.csv         # CSV file with ticker symbols and timestamps
├── requirements.txt        # List of Python dependencies
└── logs/                   # Directory for log files
    └── server.log          # Server activity log
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure the CSV file is properly set up:
   - The file should be located at `data/tickers.csv`
   - It should have columns `ticker` and `timestamp`
   - Example content:
     ```
     ticker,timestamp
     AAPL,2023-03-01 12:00:00
     MSFT,2023-03-01 12:00:00
     TSLA,2023-03-01 12:00:00
     ```

## Running the Server

### Development Mode

```
python app.py
```

This will start the server on `http://0.0.0.0:5001`.

### Production Mode

For production, use Gunicorn:

```
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

- `-w 4`: 4 worker processes for concurrent requests
- `-b 0.0.0.0:5001`: Bind to all interfaces on port 5001

## API Endpoints

### Get Options Data

```
GET /api/options/<ticker>
```

Returns options data for the specified ticker symbol.

Example response:
```json
{
  "options": {
    "2023-03-17": {
      "calls": [...],
      "puts": [...]
    },
    "2023-03-24": {
      "calls": [...],
      "puts": [...]
    }
  },
  "price": 150.25,
  "ticker": "AAPL"
}
```

### Get Available Tickers

```
GET /api/tickers
```

Returns a list of all ticker symbols available in the system.

Example response:
```json
["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]
```

### Health Check

```
GET /health
```

Returns the health status of the server.

Example response:
```json
{
  "status": "healthy"
}
```

## Maintenance

- **Logs**: Check `logs/server.log` for server activity and errors
- **Cache**: The cache is automatically updated every 10 minutes
- **Adding Tickers**: Add new tickers to `data/tickers.csv` with a timestamp 
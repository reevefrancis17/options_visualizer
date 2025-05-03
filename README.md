# Options Chain Visualizer

A web application for visualizing stock options data fetched from Yahoo Finance. This tool allows you to:
- View call and put options prices across different strike prices
- Toggle between different metrics (Price, Delta, Gamma, Theta, IV, etc.)
- Navigate through different expiration dates
- Track values at specific strike prices with interactive crosshairs
- Compare intrinsic and extrinsic option values
- Automatically track and cache frequently accessed tickers

## Project Structure

```
options_visualizer/
├── python/           # Core Python modules
├── frontend/         # Frontend code
├── backend/          # Backend
├── tests/            # Test files
├── data/             # Data files
├── cache/            # Cache files
├── main.py           # Main entry point
└── README.md         # This file
```

## Ticker Registry and Caching System

The application includes a sophisticated ticker registry and caching system:

- **Persistent Ticker Registry**: All tickers ever accessed are stored in `data/ticker_registry.json`
  - The registry only grows over time, never removing tickers
  - Tracks first access time, last access time, and access count for each ticker
  
- **Intelligent Caching**:
  - Options data is cached to reduce API calls and improve performance
  - Cache automatically refreshes stale data in the background
  - Bidirectional synchronization between cache and registry ensures consistency
  
- **Auto-Loading**:
  - Tickers in the registry are automatically loaded during refresh cycles
  - New tickers are added to the registry when first accessed
  - This creates an ever-growing library of options data for frequently accessed tickers

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

## Developer Setup

1. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python options_visualizer_backend/main.py
   ```

5. **Code Style and Linting**
   - Format code with Black:
     ```bash
     black .
     ```
   - Check code style with Flake8:
     ```bash
     flake8 .
     ```

## Running the Application

### Using the Combined App

```bash
python main.py
```

This will start both the backend and frontend together and open the web interface in your browser.

### Running Just the Backend

```bash
python -m backend.app
```

### Running with Environment Variables

You can configure the application using environment variables:

```bash
PORT=5003 python -m backend.app
```

## Using the Web Application

- Enter a stock symbol in the search box and click "Search"
- The application will fetch and display options data for the symbol
- Use the expiration date navigation buttons to switch between different expiration dates
- The current stock price is shown as a vertical green dashed line
- Calls are shown in blue, puts in red
- Data loads progressively, allowing you to see partial results while fetching continues
- Select different metrics using the radio buttons to view different aspects of the options data

## Available Metrics

The application supports visualizing various options metrics:
- **Price**: Option mid price (average of bid and ask)
- **Delta**: Rate of change of option price with respect to underlying price
- **Gamma**: Rate of change of delta with respect to underlying price
- **Theta**: Rate of change of option price with respect to time
- **IV**: Implied volatility
- **Volume**: Trading volume
- **Spread**: Difference between bid and ask prices
- **Intrinsic Value**: Value if exercised immediately
- **Extrinsic Value**: Time value and volatility premium

## Data Source

This application fetches real-time options data from Yahoo Finance. The data includes:
- Strike prices
- Bid/Ask prices
- Last price
- Volume
- Calculated Greeks (Delta, Gamma, Theta)
- Implied Volatility

## Troubleshooting

If you encounter any issues:

1. **Port Conflicts**: If either port 5001 or 5002 is already in use, you can change them:
   ```bash
   # For backend
   PORT=5003 python -m backend.app
   
   # For frontend (update config.js to point to the new backend port)
   PORT=5004 python -m options_visualizer_web.app
   ```

2. **API Connection Issues**: Ensure both servers are running and check the console for error messages.

3. **Data Loading Issues**: Some tickers may have limited options data or may be rate-limited by Yahoo Finance.

4. **Browser Console**: Check your browser's developer console (F12) for JavaScript errors.

5. **Server Logs**: Check the terminal output of both servers for error messages.

## Development Notes

- The frontend communicates with the backend via RESTful API calls
- The backend fetches data from Yahoo Finance and processes it
- Data is cached to improve performance and reduce API calls
- The application uses Plotly.js for interactive visualizations
- The frontend is built with vanilla JavaScript for simplicity

## API Endpoints

### Backend API (port 5002)
- `GET /api/options/{ticker}`: Get options data for a specific ticker
- `GET /health`: Health check endpoint

### Frontend API (port 5001)
- `GET /`: Main web interface
- `GET /health`: Health check endpoint

## Environment Variables

You can customize the application using these environment variables:

- `PORT`: Server port (default: 5001 for frontend, 5002 for backend)
- `DEBUG`: Enable debug mode (default: True for development)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

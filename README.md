# Options Chain Visualizer

A web application for visualizing stock options data fetched from Yahoo Finance. This tool allows you to:
- View call and put options prices across different strike prices
- Toggle between different metrics (Price, Delta, Gamma, Theta, IV, etc.)
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

## Running the Application in Test Environment

The application consists of two separate components:
1. **Backend API Server**: Fetches and processes options data
2. **Frontend Web Server**: Serves the web interface

To run both components for testing:

### Step 1: Start the Backend API Server
```bash
python -m options_visualizer_backend.app
```
This will start the backend server on port 5002 (http://localhost:5002).

### Step 2: Start the Frontend Web Server
```bash
python -m options_visualizer_web.app
```
This will start the frontend server on port 5001 (http://localhost:5001).

### Step 3: Access the Web Application
Open your browser and navigate to:
```
http://localhost:5001
```

## Project Structure

The project is organized into two main components:

- `options_visualizer_web/` - Contains the frontend web application
  - `app.py` - Flask application with routes and frontend API endpoints
  - `static/` - JavaScript, CSS, and other static assets
  - `templates/` - HTML templates
- `options_visualizer_backend/` - Contains the backend API server
  - `app.py` - Flask application with API endpoints for options data
  - `models/` - Contains options pricing models
  - `data/` - Data storage directory
  - `cache/` - Cache storage directory
- `python/` - Shared Python modules
  - `options_data.py` - Core options data processing logic
  - `yahoo_finance.py` - Yahoo Finance API integration
  - `cache_manager.py` - Data caching system
- `main.py` - Unified entry point that runs both servers

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
   PORT=5003 python -m options_visualizer_backend.app
   
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

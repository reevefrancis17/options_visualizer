# Options Chain Visualizer

A web application for visualizing stock options data fetched from Yahoo Finance. This tool allows you to:
- View call and put options prices across different strike prices
- Toggle between different metrics (Price, Delta, Gamma, Theta, IV, etc.)
- Navigate through different expiration dates
- Track values at specific strike prices with interactive crosshairs
- Compare intrinsic and extrinsic option values

## Features

- **Real-time Data**: Fetches current options data from Yahoo Finance
- **Interactive Visualization**: Plotly.js powered charts with crosshair tracking
- **Multiple Metrics**: View various options metrics including Greeks
- **Concurrent Processing**: Multi-threaded backend for improved performance
- **Responsive Design**: Works on desktop and mobile devices
- **Data Caching**: Reduces API calls and improves performance

## Project Structure

The project is organized into two main components:

```
options_visualizer/
├── options_visualizer_backend/    # Backend API server
│   ├── app.py                     # Flask application with API endpoints
│   ├── options_data.py            # Core options data processing logic
│   ├── options_preprocessor.py    # Data preprocessing utilities
│   ├── yahoo_finance.py           # Yahoo Finance API integration
│   ├── config.py                  # Backend configuration settings
│   ├── models/                    # Options pricing models
│   │   ├── __init__.py            # Package initialization
│   │   └── black_scholes.py       # Black-Scholes model implementation
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py            # Package initialization
│   │   ├── cache_manager.py       # Data caching system
│   │   └── cache_utils.py         # Cache helper functions
│   ├── data/                      # Data storage directory
│   ├── cache/                     # Cache storage directory
│   ├── logs/                      # Log files directory
│   ├── requirements.txt           # Backend dependencies
│   └── README.md                  # Backend documentation
│
├── options_visualizer_web/        # Frontend web application
│   ├── app.py                     # Flask application with routes
│   ├── config.py                  # Frontend configuration settings
│   ├── static/                    # Static assets
│   │   ├── js/                    # JavaScript files
│   │   │   ├── main.js            # Main application logic
│   │   │   └── config.js          # Frontend configuration
│   │   └── css/                   # CSS stylesheets
│   │       └── style.css          # Main stylesheet
│   ├── templates/                 # HTML templates
│   │   └── index.html             # Main application page
│   ├── requirements.txt           # Frontend dependencies
│   └── README.md                  # Frontend documentation
│
├── main.py                        # Unified entry point for both servers
├── setup.py                       # Package installation script
├── pyproject.toml                 # Python project configuration
├── requirements.txt               # Combined project dependencies
├── Makefile                       # Development task automation
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

## Code Structure

### Backend Components

1. **OptionsDataManager** (`options_data.py`)
   - Central manager for options data handling
   - Manages data fetching, caching, and processing
   - Implements thread pool for concurrent processing
   - Provides API for accessing options data

2. **YahooFinanceAPI** (`yahoo_finance.py`)
   - Fetches options data from Yahoo Finance
   - Handles API rate limiting and error recovery
   - Processes raw data into a usable format

3. **Black-Scholes Model** (`models/black_scholes.py`)
   - Implements the Black-Scholes option pricing model
   - Calculates option prices and Greeks (Delta, Gamma, Theta, etc.)
   - Provides implied volatility calculation

4. **OptionsCache** (`utils/cache_manager.py`)
   - Manages caching of options data
   - Implements thread-safe access to cached data
   - Handles cache invalidation and refreshing

5. **Flask API** (`app.py`)
   - Provides RESTful API endpoints for options data
   - Implements concurrent request handling
   - Manages error handling and response formatting

### Frontend Components

1. **Flask Web Server** (`app.py`)
   - Serves the web interface
   - Proxies requests to the backend API
   - Handles error responses

2. **JavaScript Application** (`static/js/main.js`)
   - Implements the interactive visualization
   - Manages user interactions and UI updates
   - Handles data fetching and processing

3. **HTML/CSS** (`templates/index.html`, `static/css/style.css`)
   - Defines the user interface structure and styling
   - Implements responsive design

### Data Flow

1. User enters a ticker symbol in the web interface
2. Frontend makes API request to backend: `GET /api/options/{ticker}`
3. Backend checks if data is cached:
   - If cached and fresh: returns immediately
   - If cached but stale: returns cached data and refreshes in background
   - If not cached: fetches from Yahoo Finance
4. Backend processes data:
   - Calculates Greeks (Delta, Gamma, Theta)
   - Computes intrinsic and extrinsic values
   - Formats data for frontend consumption
5. Frontend receives data and:
   - Updates the UI with expiration dates
   - Renders the options chain visualization
   - Enables interactive features (hover, strike selection)

## Installation

1. **Install Python**
   - Visit [python.org](https://python.org/downloads)
   - Download and install Python 3.11 or newer
   - During installation, make sure to check "Add Python to PATH"

2. **Clone this Repository**
   ```bash
   git clone https://github.com/your-username/options-visualizer.git
   cd options-visualizer
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

## Developer Setup

1. **Create a Virtual Environment**
   ```bash
   # Create a virtual environment
   python -m venv .venv
   
   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install Development Dependencies**
   ```bash
   # Install the package in development mode
   pip install -e .
   
   # Install development tools
   pip install black flake8 pytest isort mypy
   ```

3. **Set Up Pre-commit Hooks (Optional)**
   ```bash
   # Install pre-commit
   pip install pre-commit
   
   # Install the git hooks
   pre-commit install
   ```

4. **Configure IDE (Optional)**
   - For VS Code, install the Python, Pylance, and Black Formatter extensions
   - For PyCharm, enable the Black formatter in Settings > Tools > Black

## Running the Application

The application consists of two separate components that can be run individually or together:

### Option 1: Run Both Components with the Unified Entry Point

```bash
python main.py
```

This will start both the backend and frontend servers. Access the application at http://localhost:5001.

### Option 2: Run Components Separately

#### Start the Backend API Server
```bash
python -m options_visualizer_backend.app
```
This will start the backend server on port 5002 (http://localhost:5002).

#### Start the Frontend Web Server
```bash
python -m options_visualizer_web.app
```
This will start the frontend server on port 5001 (http://localhost:5001).

### Option 3: Use the Makefile

```bash
# Run both servers
make run

# Run only the backend
make run-backend

# Run only the frontend
make run-frontend
```

## Development Workflow

### Code Formatting and Linting

```bash
# Format code with Black and isort
make format

# Run linting checks
make lint

# Run tests
make test
```

### Manual Formatting and Linting

```bash
# Format Python code with Black
black options_visualizer_backend options_visualizer_web

# Sort imports with isort
isort options_visualizer_backend options_visualizer_web

# Check code style with flake8
flake8 options_visualizer_backend options_visualizer_web

# Type checking with mypy
mypy options_visualizer_backend options_visualizer_web
```

### JavaScript and CSS Formatting

```bash
# Install Node.js dependencies (first time only)
npm install --save-dev eslint prettier

# Format JavaScript files
npx prettier --write options_visualizer_web/static/js/*.js

# Lint JavaScript files
npx eslint options_visualizer_web/static/js/*.js

# Format CSS files
npx prettier --write options_visualizer_web/static/css/*.css
```

## Using the Web Application

1. Open your browser and navigate to http://localhost:5001
2. Enter a stock symbol in the search box and click "Search"
3. The application will fetch and display options data for the symbol
4. Use the expiration date navigation buttons to switch between different dates
5. Select different metrics using the radio buttons to view different aspects of the options data

## Available Metrics

- **Price**: Option mid price (average of bid and ask)
- **Delta**: Rate of change of option price with respect to underlying price
- **Gamma**: Rate of change of delta with respect to underlying price
- **Theta**: Rate of change of option price with respect to time
- **IV**: Implied volatility
- **Volume**: Trading volume
- **Spread**: Difference between bid and ask prices
- **Intrinsic Value**: Value if exercised immediately
- **Extrinsic Value**: Time value and volatility premium

## API Endpoints

### Backend API (port 5002)
- `GET /api/options/{ticker}`: Get options data for a specific ticker
  - Query parameters:
    - `dte_min`: Minimum days to expiration (optional)
    - `dte_max`: Maximum days to expiration (optional)
    - `fields`: Comma-separated list of fields to include (optional)
- `GET /api/tickers`: Get list of saved tickers
- `POST /api/tickers/add`: Add a ticker to the saved list
- `POST /api/tickers/remove`: Remove a ticker from the saved list
- `GET /api/cache/stats`: Get cache statistics
- `POST /api/cache/refresh`: Refresh the cache for a ticker
- `GET /health`: Health check endpoint

### Frontend API (port 5001)
- `GET /`: Main web interface
- `GET /api/options/{ticker}`: Proxy to backend API
- `GET /health`: Health check endpoint

## Configuration

You can customize the application using these environment variables:

- `PORT`: Server port (default: 5001 for frontend, 5002 for backend)
- `DEBUG`: Enable debug mode (default: True for development)
- `CACHE_DIR`: Directory for caching data (default: ./cache)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MAX_WORKERS`: Maximum number of worker threads (default: CPU cores * 2)
- `BACKEND_URL`: URL of the backend API (default: http://localhost:5002)

## Troubleshooting

If you encounter any issues:

1. **Port Conflicts**: If either port 5001 or 5002 is already in use, you can change them using the PORT environment variable.

2. **API Connection Issues**: Ensure both servers are running and check the console for error messages.

3. **Data Loading Issues**: Some tickers may have limited options data or may be rate-limited by Yahoo Finance.

4. **Browser Console**: Check your browser's developer console (F12) for JavaScript errors.

5. **Server Logs**: Check the terminal output of both servers for error messages.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

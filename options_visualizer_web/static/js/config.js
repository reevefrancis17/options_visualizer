/**
 * Configuration for the Options Visualizer frontend
 */
const CONFIG = {
    // Backend API URL - change this if your backend is running on a different host/port
    BACKEND_URL: 'http://localhost:5001',
    
    // Frontend API endpoints
    FRONTEND_API: {
        GET_OPTIONS_DATA: '/api/get_options_data'
    },

    // Polling interval in milliseconds
    POLLING_INTERVAL: 5000,
    
    // Default ticker
    DEFAULT_TICKER: 'SPY'
}; 
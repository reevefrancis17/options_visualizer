// Options Visualizer Web App - Frontend Logic

// Field mapping for plot display (matches the Python backend)
const FIELD_MAPPING = {
    "Price": "mid_price",
    "Delta": "delta",
    "Gamma": "gamma",
    "Theta": "theta",
    "IV": "impliedVolatility",
    "Volume": "volume",
    "Spread": "spread",
    "Intrinsic Value": "intrinsic_value",
    "Extrinsic Value": "extrinsic_value"
};

// Price-related fields that need dollar formatting
const PRICE_FIELDS = ["Price", "Spread", "Intrinsic Value", "Extrinsic Value"];

// Global state
const state = {
    symbol: "",
    currentPrice: null,
    expiryDates: [],
    currentExpiryIndex: 0,
    optionsData: [],
    lastUpdateTime: null,
    plot: null,
    hoveredStrike: null,
    isPolling: false,
    pollingInterval: null,
    lastProcessedDates: 0,
    totalDates: 0,
    isHovering: false,
    plotHasMouseLeaveHandler: false
};

// DOM elements
const elements = {
    ticker: document.getElementById('ticker'),
    searchBtn: document.getElementById('search-btn'),
    statusLabel: document.getElementById('status-label'),
    expiryLabel: document.getElementById('expiry-label'),
    prevBtn: document.getElementById('prev-btn'),
    nextBtn: document.getElementById('next-btn'),
    plotOptions: document.querySelectorAll('input[name="plot-type"]'),
    plotContainer: document.getElementById('options-plot'),
    errorMessage: document.getElementById('error-message')
};

// Initialize the app
function init() {
    console.log("Initializing app...");
    
    // Check if Plotly is loaded
    if (typeof Plotly === 'undefined') {
        console.error("Plotly library is not loaded!");
        showError("Failed to load the plotting library. Please refresh the page or check your internet connection.");
        return;
    } else {
        console.log("Plotly library is loaded, version:", Plotly.version);
    }
    
    // Check if the plot container exists and has dimensions
    const plotContainer = elements.plotContainer;
    if (!plotContainer) {
        console.error("Plot container not found!");
    } else {
        console.log(`Plot container dimensions: ${plotContainer.offsetWidth}x${plotContainer.offsetHeight}`);
        if (plotContainer.offsetWidth === 0 || plotContainer.offsetHeight === 0) {
            console.warn("Plot container has zero dimensions, this may cause Plotly to fail");
        }
    }
    
    // Add event listeners
    elements.searchBtn.addEventListener('click', searchTicker);
    elements.ticker.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchTicker();
        }
    });
    elements.prevBtn.addEventListener('click', prevExpiry);
    elements.nextBtn.addEventListener('click', nextExpiry);
    
    // Add event listeners for plot type change
    elements.plotOptions.forEach(option => {
        option.addEventListener('change', updatePlot);
    });
    
    // Load default ticker (SPY)
    elements.ticker.value = CONFIG.DEFAULT_TICKER;
    console.log(`Starting initial search for ${CONFIG.DEFAULT_TICKER}...`);
    searchTicker();
}

// Search for a ticker
function searchTicker() {
    const ticker = elements.ticker.value.trim().toUpperCase();
    console.log(`Searching for ticker: ${ticker}`);
    if (!ticker || !ticker.match(/^[A-Z0-9]+$/)) {
        showError('Invalid ticker symbol. Please enter a valid stock symbol.');
        return;
    }
    
    // Update UI state
    elements.searchBtn.disabled = true;
    hideError();
    elements.statusLabel.textContent = `Loading ${ticker} data...`;
    
    // Stop any existing polling
    stopPolling();
    
    // Reset state
    state.lastProcessedDates = 0;
    state.totalDates = 0;
    
    // Fetch data from the API
    fetchOptionsData(ticker);
}

// Fetch options data with support for partial data
function fetchOptionsData(ticker, isPolling = false) {
    console.log(`Fetching options data for ${ticker}, polling: ${isPolling}`);
    
    // If this is the first request (not polling), show a loading state immediately
    if (!isPolling) {
        elements.statusLabel.textContent = `Loading ${ticker} data...`;
        elements.searchBtn.disabled = true;
        hideError(); // Clear any previous errors
        
        // Disable navigation buttons during initial load
        elements.prevBtn.disabled = true;
        elements.nextBtn.disabled = true;
    }
    
    // Log the URL we're fetching from
    const url = `${CONFIG.BACKEND_URL}/api/options/${ticker}`;
    console.log(`Fetching from URL: ${url}`);
    
    // Use the backend API directly
    fetch(url, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log(`Response status: ${response.status}`);
        if (!response.ok) {
            return response.json().then(data => {
                console.error(`Error response data:`, data);
                throw new Error(data.message || data.error || `Failed to fetch data for ${ticker}`);
            });
        }
        // Get the response text first to sanitize any NaN values
        return response.text();
    })
    .then(text => {
        // Replace NaN with null in the JSON string
        const sanitizedText = text.replace(/:\s*NaN\b/g, ': null').replace(/:\s*-NaN\b/g, ': null');
        try {
            const parsedData = JSON.parse(sanitizedText);
            // Further sanitize the parsed data to catch any NaN values that might have slipped through
            return sanitizeData(parsedData);
        } catch (error) {
            console.error('Error parsing JSON:', error);
            console.error('Response text:', text);
            console.error('Sanitized text:', sanitizedText);
            throw new Error(`Failed to parse response for ${ticker}: ${error.message}`);
        }
    })
    .then(data => {
        console.log(`Received data for ${ticker}:`, data);
        
        // Update UI state
        elements.searchBtn.disabled = false;
        
        // Handle loading state
        if (data.status === 'loading' || data.status === 'partial') {
            // Ensure progress is a valid number between 0 and 1
            const progress = typeof data.progress === 'number' && !isNaN(data.progress) 
                ? Math.max(0, Math.min(1, data.progress)) 
                : 0;
            
            const progressPercent = Math.round(progress * 100);
            const progressText = progressPercent > 0 ? `(${progressPercent}%)` : '';
            
            const statusMessage = data.status === 'loading' 
                ? `Loading ${ticker} data... ${progressText}` 
                : `Partial data for ${ticker} ${progressText}`;
            
            console.log(`Data for ${ticker} is still loading/partial, progress: ${progress}`);
            elements.statusLabel.textContent = statusMessage;
            
            // Show a friendly message in the error area (not as an error)
            const loadingMessage = data.status === 'loading'
                ? `First time loading ${ticker}. Please wait while we fetch and process the data...`
                : `Loading more data for ${ticker}. Partial data is displayed.`;
            
            showError(loadingMessage, true);
            
            // Start polling if not already polling
            if (!isPolling) {
                startPolling(ticker, data.status === 'loading');
            }
            
            // If we have partial data, we can still display it
            if (data.status === 'partial' && data.options_data && data.options_data.length > 0) {
                // Process the partial data
                processReceivedData(data, ticker);
            }
            
            return;
        }
        
        // Hide any info messages
        elements.errorMessage.style.display = 'none';
        
        // Handle error state
        if (data.status === 'error') {
            console.error(`Error fetching data for ${ticker}:`, data.message || data.error);
            showError(data.message || data.error || `Failed to fetch data for ${ticker}`);
            stopPolling();
            elements.statusLabel.textContent = `Error: ${data.message || 'Failed to fetch data'}`;
            return;
        }
        
        // Stop polling if we were polling
        if (isPolling) {
            stopPolling();
        }
        
        // Process the complete data
        processReceivedData(data, ticker);
    })
    .catch(error => {
        console.error(`Error fetching data for ${ticker}:`, error);
        elements.searchBtn.disabled = false;
        
        // Check if the error message indicates the data is still being processed
        if (error.message && (
            error.message.includes("Unknown error occurred") || 
            error.message.includes("still loading") ||
            error.message.includes("processing") ||
            error.message.includes("fetching")
        )) {
            // This is likely a new ticker being processed for the first time
            elements.statusLabel.textContent = `Loading ${ticker} data... Please wait.`;
            
            // Show a friendly message
            const loadingMessage = `First time loading ${ticker}. Please wait while we fetch and process the data...`;
            showError(loadingMessage, true);
            
            // Start polling to check when data becomes available
            if (!isPolling) {
                startPolling(ticker, true);
            }
        } else {
            // This is a real error
            elements.statusLabel.textContent = `Error loading ${ticker} data`;
            showError(error.message);
            stopPolling();
        }
    });
}

// Helper function to sanitize data by replacing NaN values with null
function sanitizeData(obj) {
    if (obj === null || obj === undefined) {
        return obj;
    }
    
    if (typeof obj === 'number' && isNaN(obj)) {
        return null;
    }
    
    if (Array.isArray(obj)) {
        return obj.map(item => sanitizeData(item));
    }
    
    if (typeof obj === 'object') {
        const result = {};
        for (const key in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, key)) {
                result[key] = sanitizeData(obj[key]);
            }
        }
        return result;
    }
    
    return obj;
}

// Process received data from the API
function processReceivedData(data, ticker) {
    // Process the data
    state.symbol = ticker;
    state.currentPrice = data.current_price;
    
    // Format expiry dates consistently
    state.expiryDates = (data.expiry_dates || []).map(date => {
        // Check if the date is already a string in YYYY-MM-DD format
        if (typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date)) {
            return date;
        }
        // If it's a timestamp or other format, convert to YYYY-MM-DD
        return new Date(date).toISOString().split('T')[0];
    });
    
    // Process options data to ensure consistent date format
    state.optionsData = (data.options_data || []).map(item => {
        // Create a new object to avoid modifying the original
        const newItem = {...item};
        
        // Ensure expiration is in YYYY-MM-DD format
        if (newItem.expiration) {
            if (typeof newItem.expiration === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(newItem.expiration)) {
                // Already in the correct format
            } else {
                // Convert to YYYY-MM-DD
                newItem.expiration = new Date(newItem.expiration).toISOString().split('T')[0];
            }
        }
        
        return newItem;
    });
    
    state.lastUpdateTime = new Date();
    state.lastProcessedDates = data.processed_dates || 0;
    state.totalDates = data.total_dates || 0;
    
    console.log(`Processed data for ${ticker}:`, {
        currentPrice: state.currentPrice,
        expiryDates: state.expiryDates.length,
        optionsData: state.optionsData.length,
        lastProcessedDates: state.lastProcessedDates,
        totalDates: state.totalDates
    });
    
    // Update status label
    if (data.status === 'partial') {
        // Calculate percentage, ensuring it's a valid number
        let percent = 0;
        if (state.totalDates > 0 && !isNaN(state.lastProcessedDates) && !isNaN(state.totalDates)) {
            percent = Math.round((state.lastProcessedDates / state.totalDates) * 100);
            // Ensure percent is a valid number between 0 and 100
            percent = isNaN(percent) ? 0 : Math.max(0, Math.min(100, percent));
        }
        elements.statusLabel.textContent = `Partial data for ${ticker} (${percent}% complete)`;
    } else {
        elements.statusLabel.textContent = `${ticker} data updated at ${state.lastUpdateTime.toLocaleTimeString()}`;
    }
    
    // Reset expiry index if needed
    if (state.currentExpiryIndex >= state.expiryDates.length) {
        state.currentExpiryIndex = 0;
    }
    
    // Update navigation buttons
    updateExpiryDisplay();
    
    // Update the plot
    updatePlot();
}

// Start polling for updates
function startPolling(ticker, isNewTicker = false) {
    console.log(`Starting polling for ${ticker}, isNewTicker: ${isNewTicker}`);
    state.isPolling = true;
    
    // Clear any existing polling interval
    if (state.pollingInterval) {
        clearInterval(state.pollingInterval);
    }
    
    // Use a more frequent polling interval for new tickers (2 seconds instead of 5)
    const pollingInterval = isNewTicker ? 2000 : CONFIG.POLLING_INTERVAL;
    
    // Set up new polling interval
    state.pollingInterval = setInterval(() => {
        console.log(`Polling for ${ticker} updates...`);
        fetchOptionsData(ticker, true);
    }, pollingInterval);
}

// Stop polling for updates
function stopPolling() {
    console.log("Stopping polling");
    state.isPolling = false;
    
    if (state.pollingInterval) {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
    }
}

// Update the expiry date display and navigation buttons
function updateExpiryDisplay() {
    console.log("Updating expiry display");
    if (!state.expiryDates || state.expiryDates.length === 0) {
        elements.expiryLabel.textContent = 'No data';
        elements.prevBtn.disabled = true;
        elements.nextBtn.disabled = true;
        return;
    }
    
    // Get current date and update label
    const currentDate = state.expiryDates[state.currentExpiryIndex];
    elements.expiryLabel.textContent = currentDate;
    
    // Update navigation buttons
    elements.prevBtn.disabled = state.currentExpiryIndex <= 0;
    elements.nextBtn.disabled = state.currentExpiryIndex >= state.expiryDates.length - 1;
}

// Navigate to previous expiry date
function prevExpiry() {
    if (state.currentExpiryIndex > 0) {
        state.currentExpiryIndex--;
        updateExpiryDisplay();
        updatePlot();
    }
}

// Navigate to next expiry date
function nextExpiry() {
    if (state.currentExpiryIndex < state.expiryDates.length - 1) {
        state.currentExpiryIndex++;
        updateExpiryDisplay();
        updatePlot();
    }
}

// Update the plot with current data
function updatePlot() {
    try {
        console.log("Updating plot with current data");
        console.log("State:", {
            optionsData: state.optionsData ? state.optionsData.length : 0,
            expiryDates: state.expiryDates ? state.expiryDates.length : 0,
            currentExpiryIndex: state.currentExpiryIndex,
            currentPrice: state.currentPrice
        });
        
        if (!state.optionsData || state.optionsData.length === 0 || !state.expiryDates || state.expiryDates.length === 0) {
            console.warn("No data to plot");
            return;
        }
        
        const selectedPlotType = Array.from(elements.plotOptions).find(option => option.checked).value;
        const plotField = FIELD_MAPPING[selectedPlotType];
        console.log(`Selected plot type: ${selectedPlotType}, field: ${plotField}`);
        
        // Get current date
        const currentDate = state.expiryDates[state.currentExpiryIndex];
        console.log(`Current expiry date: ${currentDate}`);
        
        // Filter data for current expiry date
        const filteredData = state.optionsData.filter(item => item.expiration === currentDate);
        console.log(`Filtered data for ${currentDate}: ${filteredData.length} items`);
        
        // Separate calls and puts
        const calls = filteredData.filter(item => item.option_type === 'call');
        const puts = filteredData.filter(item => item.option_type === 'put');
        console.log(`Calls: ${calls.length}, Puts: ${puts.length}`);
        
        // Sort by strike price
        calls.sort((a, b) => a.strike - b.strike);
        puts.sort((a, b) => a.strike - b.strike);
        
        // Process data based on plot type
        let callValues = calls.map(item => item[plotField]);
        let putValues = puts.map(item => item[plotField]);
        
        // Check for missing or invalid values
        const callNulls = callValues.filter(v => v === null || v === undefined || isNaN(v)).length;
        const putNulls = putValues.filter(v => v === null || v === undefined || isNaN(v)).length;
        console.log(`Call values: ${callValues.length} (${callNulls} null/undefined/NaN)`);
        console.log(`Put values: ${putValues.length} (${putNulls} null/undefined/NaN)`);
        
        // Special handling for IV (convert to percentage)
        if (selectedPlotType === 'IV') {
            callValues = callValues.map(val => val !== null ? val * 100 : null);
            putValues = putValues.map(val => val !== null ? val * 100 : null);
        }
        
        // Prepare data for plotting - always use 'lines' mode for all plot types
        const callTrace = {
            x: calls.map(item => item.strike),
            y: callValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Calls',
            line: {
                color: 'blue',
                width: 2
            }
        };
        
        const putTrace = {
            x: puts.map(item => item.strike),
            y: putValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Puts',
            line: {
                color: 'red',
                width: 2
            }
        };
        
        // Add current price line
        const currentPriceLine = {
            x: [state.currentPrice, state.currentPrice],
            y: [0, Math.max(...callValues.filter(v => v !== null && !isNaN(v) && v !== undefined), 
                             ...putValues.filter(v => v !== null && !isNaN(v) && v !== undefined)) * 1.1 || 1],
            type: 'scatter',
            mode: 'lines',
            name: `Spot: $${state.currentPrice.toFixed(2)}`, // Changed from "Current Price"
            line: {
                color: 'green',
                width: 2,
                dash: 'dash'
            }
        };
        
        // Add hover line (initially hidden)
        const hoverLine = {
            x: [0, 0],
            y: [0, 0],
            type: 'scatter',
            mode: 'lines',
            name: 'Strike',
            line: {
                color: 'gray',
                width: 1,
                dash: 'dot'
            },
            hoverinfo: 'none',
            visible: false
        };
        
        // Layout configuration
        const layout = {
            title: `${state.symbol} Options - ${selectedPlotType} (${currentDate})`,
            xaxis: {
                title: 'Strike Price ($)',
                tickprefix: '$'
            },
            yaxis: {
                title: getYAxisTitle(selectedPlotType),
                tickprefix: PRICE_FIELDS.includes(selectedPlotType) ? '$' : '',
                tickformat: getTickFormat(selectedPlotType)
            },
            hovermode: 'closest',
            showlegend: true,
            legend: {
                x: 0,
                y: 1
            },
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4
            }
        };
        
        // Create or update the plot
        const plotData = [callTrace, putTrace, currentPriceLine, hoverLine];

        if (!state.plot) {
            // Create new plot
            Plotly.newPlot('options-plot', plotData, layout, {
                responsive: true,
                displayModeBar: false
            })
                .then(plot => {
                    state.plot = plot;
                    
                    // Add hover event
                    plot.on('plotly_hover', function(data) {
                        const strike = data.points[0].x;
                        state.hoveredStrike = strike;
                        updateHoverLine(strike);
                    });
                    
                    // Add unhover event
                    plot.on('plotly_unhover', function() {
                        const atmStrike = findAtmStrike(calls, puts, state.currentPrice);
                        updateHoverLine(atmStrike);
                    });
                    
                    // Initialize hover line at ATM strike
                    const atmStrike = findAtmStrike(calls, puts, state.currentPrice);
                    updateHoverLine(atmStrike);
                })
                .catch(err => {
                    console.error("Error creating plot:", err);
                    showError("Failed to create the plot. Please try again.");
                });
        } else {
            // Update existing plot
            Plotly.react('options-plot', plotData, layout, {
                responsive: true,
                displayModeBar: false
            })
                .then(() => {
                    // Update hover line to ATM strike after plot update
                    const atmStrike = findAtmStrike(calls, puts, state.currentPrice);
                    updateHoverLine(atmStrike);
                })
                .catch(err => {
                    console.error("Error updating plot:", err);
                    showError("Failed to update the plot. Please try again.");
                });
        }
    } catch (error) {
        console.error("Error updating plot:", error);
        showError(`Failed to update the plot: ${error.message}`);
    }
}

// Helper function to find the value at a specific strike
function findValueAtStrike(data, strike, field) {
    // Filter out null, undefined and NaN values first
    const validData = data.filter(item => 
        item.strike !== null && 
        item.strike !== undefined && 
        !isNaN(item.strike) && 
        item[field] !== null && 
        item[field] !== undefined && 
        !isNaN(item[field])
    );
    
    if (validData.length === 0) {
        return null;
    }
    
    // Find the exact match first
    const exactMatch = validData.find(item => item.strike === strike);
    if (exactMatch) {
        return exactMatch[field];
    }
    
    // Sort the data by strike price for proper interpolation
    validData.sort((a, b) => a.strike - b.strike);
    
    // Find the closest strikes (one below and one above)
    let lowerIndex = -1;
    for (let i = 0; i < validData.length; i++) {
        if (validData[i].strike <= strike) {
            lowerIndex = i;
        } else {
            break;
        }
    }
    
    // If we're at the edges of our data, return the closest point
    if (lowerIndex === -1) {
        return validData[0][field]; // Below the lowest strike
    }
    if (lowerIndex === validData.length - 1) {
        return validData[lowerIndex][field]; // Above the highest strike
    }
    
    // We have points on both sides, interpolate
    const lowerStrike = validData[lowerIndex].strike;
    const upperStrike = validData[lowerIndex + 1].strike;
    const lowerValue = validData[lowerIndex][field];
    const upperValue = validData[lowerIndex + 1][field];
    
    // Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    const ratio = (strike - lowerStrike) / (upperStrike - lowerStrike);
    const interpolatedValue = lowerValue + ratio * (upperValue - lowerValue);
    
    return interpolatedValue;
}

// Find the closest strike to the current price (at-the-money)
function findAtmStrike(calls, puts, currentPrice) {
    // Combine all strikes from calls and puts
    const allStrikes = [...new Set([
        ...calls.map(item => item.strike),
        ...puts.map(item => item.strike)
    ])].filter(strike => strike !== null && strike !== undefined && !isNaN(strike));
    
    if (allStrikes.length === 0) {
        return currentPrice; // Default to current price if no strikes available
    }
    
    // Sort strikes
    allStrikes.sort((a, b) => a - b);
    
    // Find the closest strike to current price
    let closestStrike = allStrikes[0];
    let minDiff = Math.abs(currentPrice - closestStrike);
    
    for (let i = 1; i < allStrikes.length; i++) {
        const diff = Math.abs(currentPrice - allStrikes[i]);
        if (diff < minDiff) {
            minDiff = diff;
            closestStrike = allStrikes[i];
        }
    }
    
    return closestStrike;
}

// Show error message
function showError(message, isInfo = false) {
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
        elements.errorMessage.style.display = 'block';
        
        // Set the appropriate class based on whether this is an error or info message
        if (isInfo) {
            elements.errorMessage.className = 'info-message';
        } else {
            elements.errorMessage.className = ''; // Reset to default (error style)
        }
    }
}

// Hide error message
function hideError() {
    if (elements.errorMessage) {
        elements.errorMessage.style.display = 'none';
        elements.errorMessage.textContent = '';
    }
}

// Update hover line
function updateHoverLine(strike) {
    try {
        const plotDiv = document.getElementById('options-plot');
        if (!plotDiv || !plotDiv.data) {
            return;
        }
        
        const plotData = plotDiv.data;
        const selectedPlotType = Array.from(elements.plotOptions).find(option => option.checked).value;
        const plotField = FIELD_MAPPING[selectedPlotType];
        
        // Get current date
        const currentDate = state.expiryDates[state.currentExpiryIndex];
        
        // Filter data for current expiry date
        const filteredData = state.optionsData.filter(item => item.expiration === currentDate);
        
        // Separate calls and puts
        const calls = filteredData.filter(item => item.option_type === 'call');
        const puts = filteredData.filter(item => item.option_type === 'put');
        
        // Find the call and put values at this strike
        const callData = calls.find(item => Math.abs(item.strike - strike) < 0.01);
        const putData = puts.find(item => Math.abs(item.strike - strike) < 0.01);
        
        // Format values based on plot type
        function formatValue(value) {
            if (value == null || isNaN(value)) return 'N/A';
            
            switch(selectedPlotType) {
                case 'IV':
                    return `${(value * 100).toFixed(1)}%`;
                case 'Delta':
                case 'Gamma':
                case 'Theta':
                    return value.toFixed(4);
                case 'Volume':
                    return value.toLocaleString();
                case 'Price':
                case 'Spread':
                case 'Intrinsic Value':
                case 'Extrinsic Value':
                    return `$${value.toFixed(2)}`;
                default:
                    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
            }
        }
        
        // Update trace names with values
        if (callData) {
            plotData[0].name = `Calls: ${formatValue(callData[plotField])}`;
        } else {
            plotData[0].name = 'Calls: N/A';
        }
        
        if (putData) {
            plotData[1].name = `Puts: ${formatValue(putData[plotField])}`;
        } else {
            plotData[1].name = 'Puts: N/A';
        }
        
        // Update strike line
        plotData[3].name = `Strike: $${strike.toFixed(2)}`;
        plotData[3].x = [strike, strike];
        plotData[3].visible = true;
        
        // Get y-axis range
        let yValues = [];
        if (selectedPlotType === 'IV') {
            // For IV, we need to multiply by 100 to match the display
            yValues = [...calls.map(item => item[plotField] * 100), ...puts.map(item => item[plotField] * 100)]
                .filter(y => y != null && !isNaN(y));
        } else {
            yValues = [...calls.map(item => item[plotField]), ...puts.map(item => item[plotField])]
                .filter(y => y != null && !isNaN(y));
        }
        
        const yMax = Math.max(...yValues) * 1.1 || 1;
        plotData[3].y = [0, yMax];
        
        // Track if we're currently hovering
        state.isHovering = strike !== findAtmStrike(calls, puts, state.currentPrice);
        
        Plotly.redraw('options-plot')
            .catch(err => {
                console.error("Error redrawing plot:", err);
            });
    } catch (error) {
        console.error('Error updating hover line:', error);
    }
}

// Helper functions for plot formatting
function getYAxisTitle(plotType) {
    switch(plotType) {
        case 'Price':
            return 'Option Price ($)';
        case 'Delta':
            return 'Delta (Δ)';
        case 'Gamma':
            return 'Gamma (Γ)';
        case 'Theta':
            return 'Theta (Θ) - Daily';
        case 'IV':
            return 'Implied Volatility (%)';
        case 'Volume':
            return 'Trading Volume';
        case 'Spread':
            return 'Bid-Ask Spread ($)';
        case 'Intrinsic Value':
            return 'Intrinsic Value ($)';
        case 'Extrinsic Value':
            return 'Extrinsic Value ($)';
        default:
            return plotType;
    }
}

function getTickFormat(plotType) {
    switch(plotType) {
        case 'IV':
            return '.1%';  // Format as percentage with 1 decimal place
        case 'Delta':
        case 'Gamma':
        case 'Theta':
            return '.4f';  // Format with 4 decimal places
        case 'Volume':
            return ',d';   // Format as integer with commas
        case 'Price':
        case 'Spread':
        case 'Intrinsic Value':
        case 'Extrinsic Value':
            return '.2f';  // Format with 2 decimal places
        default:
            return '';
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM content loaded, initializing app");
    init();
    
    // Add window resize handler to make plot responsive
    window.addEventListener('resize', function() {
        if (state.optionsData && state.optionsData.length > 0 && state.plot) {
            try {
                Plotly.relayout('options-plot', {
                    autosize: true,
                    displayModeBar: false
                }).catch(err => {
                    console.error("Error resizing plot:", err);
                });
            } catch (err) {
                console.error("Exception during plot resize:", err);
            }
        }
    });
}); 
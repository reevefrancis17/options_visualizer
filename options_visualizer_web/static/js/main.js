// Options Visualizer Web App - Frontend Logic

// Field mapping for plot display (matches the Python backend)
const FIELD_MAPPING = {
    "Price": "mid_price",
    "Bid": "bid",
    "Ask": "ask",
    "Volume": "volume",
    "Intrinsic Value": "intrinsic_value",
    "Extrinsic Value": "extrinsic_value"
};

// Price-related fields that need dollar formatting
const PRICE_FIELDS = ["Price", "Bid", "Ask", "Intrinsic Value", "Extrinsic Value"];

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
    elements.ticker.value = 'SPY';
    console.log("Starting initial search for SPY...");
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
    
    // Update UI state
    elements.searchBtn.disabled = true;
    if (!isPolling) {
        elements.statusLabel.textContent = `Loading ${ticker} data...`;
        hideError();
    }
    
    // Create a timeout promise
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Request timed out')), 30000); // 30 second timeout
    });
    
    // Use the backend API directly with timeout
    Promise.race([
        fetch(`${CONFIG.BACKEND_URL}/api/options/${ticker}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        }),
        timeoutPromise
    ])
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || `Failed to fetch data for ${ticker}`);
            }).catch(e => {
                // If JSON parsing fails, throw the original HTTP error
                throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log(`Received data for ${ticker}:`, data);
        
        // Update UI state
        elements.searchBtn.disabled = false;
        
        if (data.status === 'loading') {
            // Data is still loading, start polling
            elements.statusLabel.textContent = `Loading ${ticker} data... (${Math.round(data.progress || 0)}%)`;
            if (!isPolling) {
                startPolling(ticker);
            }
            return;
        }
        
        // Stop polling if we were polling
        if (isPolling) {
            stopPolling();
        }
        
        // Process the data
        state.symbol = ticker;
        state.currentPrice = data.current_price;
        state.expiryDates = data.expiry_dates || [];
        state.optionsData = data.options_data || [];
        state.lastUpdateTime = new Date();
        state.lastProcessedDates = data.processed_dates || 0;
        state.totalDates = data.total_dates || 0;
        
        // Update status label
        if (data.status === 'partial') {
            const percent = Math.round((state.lastProcessedDates / state.totalDates) * 100);
            elements.statusLabel.textContent = `Partial data for ${ticker} (${percent}% complete)`;
            
            // Start polling for updates if not already polling
            if (!isPolling) {
                startPolling(ticker);
            }
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
    })
    .catch(error => {
        console.error(`Error fetching data for ${ticker}:`, error);
        elements.searchBtn.disabled = false;
        elements.statusLabel.textContent = `Error loading ${ticker} data`;
        showError(error.message || 'Network error occurred');
        
        // Don't stop polling on first error if we're already polling
        if (!isPolling) {
            startPolling(ticker); // Start polling to retry
        }
    });
}

// Start polling for more data
function startPolling(ticker) {
    state.isPolling = true;
    console.log(`Starting polling for ${ticker}`);
    
    // Clear any existing interval
    if (state.pollingInterval) {
        clearInterval(state.pollingInterval);
    }
    
    // Track retry attempts and backoff
    let retryCount = 0;
    const maxRetries = 5;
    const baseDelay = 2000; // Start with 2 seconds
    
    // Poll with exponential backoff on errors
    state.pollingInterval = setInterval(() => {
        // If we've loaded all dates, stop polling
        if (state.lastProcessedDates >= state.totalDates && state.totalDates > 0) {
            console.log(`Polling complete for ${ticker}: all ${state.totalDates} dates processed`);
            stopPolling();
            return;
        }
        
        // If we've exceeded max retries, stop polling
        if (retryCount >= maxRetries) {
            console.error(`Exceeded maximum retry attempts (${maxRetries}) for ${ticker}`);
            stopPolling();
            showError(`Failed to load complete data for ${ticker} after multiple attempts. Please try again later.`);
            return;
        }
        
        // Fetch data with error handling
        fetch(`${CONFIG.BACKEND_URL}/api/options/${ticker}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || `Failed to fetch data for ${ticker}`);
                });
            }
            // Reset retry count on successful response
            retryCount = 0;
            return response.json();
        })
        .then(data => {
            console.log(`Polling received data for ${ticker}:`, data);
            
            if (data.status === 'loading') {
                // Data is still loading
                elements.statusLabel.textContent = `Loading ${ticker} data... (${Math.round(data.progress || 0)}%)`;
                return;
            }
            
            // Process the data
            state.symbol = ticker;
            state.currentPrice = data.current_price;
            state.expiryDates = data.expiry_dates || [];
            state.optionsData = data.options_data || [];
            state.lastUpdateTime = new Date();
            state.lastProcessedDates = data.processed_dates || 0;
            state.totalDates = data.total_dates || 0;
            
            // Update status label
            if (data.status === 'partial') {
                const percent = Math.round((state.lastProcessedDates / state.totalDates) * 100);
                elements.statusLabel.textContent = `Partial data for ${ticker} (${percent}% complete)`;
            } else {
                elements.statusLabel.textContent = `${ticker} data updated at ${state.lastUpdateTime.toLocaleTimeString()}`;
                stopPolling(); // Stop polling if we have complete data
            }
            
            // Reset expiry index if needed
            if (state.currentExpiryIndex >= state.expiryDates.length) {
                state.currentExpiryIndex = 0;
            }
            
            // Update navigation buttons
            updateExpiryDisplay();
            
            // Update the plot
            updatePlot();
        })
        .catch(error => {
            console.error(`Error during polling for ${ticker}:`, error);
            retryCount++;
            
            // Update status to show retry attempt
            elements.statusLabel.textContent = `Error loading ${ticker} data. Retry ${retryCount}/${maxRetries}...`;
            
            // Don't show error message for retries to avoid UI clutter
            if (retryCount >= maxRetries) {
                showError(`Failed to load data for ${ticker}: ${error.message}`);
            }
        });
    }, baseDelay);
}

// Stop polling
function stopPolling() {
    console.log("Stopping polling");
    state.isPolling = false;
    
    if (state.pollingInterval) {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
    }
    
    elements.searchBtn.disabled = false;
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
        if (!state.optionsData || state.optionsData.length === 0 || !state.expiryDates || state.expiryDates.length === 0) {
            console.warn("No data to plot");
            return;
        }

        const selectedPlotType = Array.from(elements.plotOptions).find(option => option.checked).value;
        const plotField = FIELD_MAPPING[selectedPlotType];
        const currentDate = state.expiryDates[state.currentExpiryIndex];
        const filteredData = state.optionsData.filter(item => item.expiration === currentDate);
        const calls = filteredData.filter(item => item.option_type === 'call').sort((a, b) => a.strike - b.strike);
        const puts = filteredData.filter(item => item.option_type === 'put').sort((a, b) => a.strike - b.strike);

        // Calculate ATM strike
        const strikes = [...new Set(filteredData.map(item => item.strike))].sort((a, b) => a - b);
        const atm_strike = strikes.reduce((prev, curr) => 
            Math.abs(curr - state.currentPrice) < Math.abs(prev - state.currentPrice) ? curr : prev, strikes[0]);

        // Calculate yMax
        const yValues = [...calls.map(item => item[plotField]), ...puts.map(item => item[plotField])]
            .filter(y => y != null && !isNaN(y));
        const yMax = yValues.length > 0 ? Math.max(...yValues) * 1.1 : 1;

        // Get call and put values at ATM strike
        const callData = calls.find(item => Math.abs(item.strike - atm_strike) < 0.01);
        const putData = puts.find(item => Math.abs(item.strike - atm_strike) < 0.01);
        const callValue = callData && callData[plotField] != null && !isNaN(callData[plotField]) ?
            (PRICE_FIELDS.includes(selectedPlotType) ? `$${callData[plotField].toFixed(2)}` :
            (Number.isInteger(callData[plotField]) ? callData[plotField].toString() : callData[plotField].toFixed(2))) : 'N/A';
        const putValue = putData && putData[plotField] != null && !isNaN(putData[plotField]) ?
            (PRICE_FIELDS.includes(selectedPlotType) ? `$${putData[plotField].toFixed(2)}` :
            (Number.isInteger(putData[plotField]) ? putData[plotField].toString() : putData[plotField].toFixed(2))) : 'N/A';

        // Define traces with initial values
        const callTrace = {
            x: calls.map(item => item.strike),
            y: calls.map(item => item[plotField]),
            type: 'scatter',
            mode: 'lines+markers',
            name: `Calls: ${callValue}`,
            line: { color: 'blue', width: 2 },
            marker: { size: 6, color: 'blue' }
        };

        const putTrace = {
            x: puts.map(item => item.strike),
            y: puts.map(item => item[plotField]),
            type: 'scatter',
            mode: 'lines+markers',
            name: `Puts: ${putValue}`,
            line: { color: 'red', width: 2 },
            marker: { size: 6, color: 'red' }
        };

        // Add current price line
        const currentPriceLine = {
            x: [state.currentPrice, state.currentPrice],
            y: [0, Math.max(...calls.map(item => item[plotField] || 0), ...puts.map(item => item[plotField] || 0)) * 1.1],
            type: 'scatter',
            mode: 'lines',
            name: `Spot: $${state.currentPrice.toFixed(2)}`, // Changed from "Current Price"
            line: {
                color: 'green',
                width: 2,
                dash: 'dash'
            }
        };

        const hoverLine = {
            x: [atm_strike, atm_strike],
            y: [0, yMax],
            type: 'scatter',
            mode: 'lines',
            name: `Strike: $${atm_strike.toFixed(2)}`,
            line: { color: 'gray', width: 1, dash: 'dot' },
            hoverinfo: 'none',
            visible: true // Initially visible at ATM
        };

        const layout = {
            title: `${state.symbol} Options - ${selectedPlotType} (${currentDate})`,
            xaxis: { title: 'Strike Price ($)', tickprefix: '$' },
            yaxis: {
                title: PRICE_FIELDS.includes(selectedPlotType) ? `${selectedPlotType} ($)` : selectedPlotType,
                tickprefix: PRICE_FIELDS.includes(selectedPlotType) ? '$' : ''
            },
            hovermode: 'closest',
            showlegend: true,
            legend: { x: 0, y: 1 },
            margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
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
function showError(message) {
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
        elements.errorMessage.style.display = 'block';
    }
}

// Hide error message
function hideError() {
    if (elements.errorMessage) {
        elements.errorMessage.style.display = 'none';
    }
}

// Update hover line and information
function updateHoverLine(strike) {
    try {
        const plotDiv = document.getElementById('options-plot');
        if (!plotDiv || !plotDiv.data) return;

        const selectedPlotType = Array.from(elements.plotOptions).find(option => option.checked).value;
        const plotField = FIELD_MAPPING[selectedPlotType];
        const currentDate = state.expiryDates[state.currentExpiryIndex];
        const filteredData = state.optionsData.filter(item => item.expiration === currentDate);
        const calls = filteredData.filter(item => item.option_type === 'call');
        const puts = filteredData.filter(item => item.option_type === 'put');

        const callData = calls.find(item => Math.abs(item.strike - strike) < 0.01);
        const putData = puts.find(item => Math.abs(item.strike - strike) < 0.01);
        const callValue = callData && callData[plotField] != null && !isNaN(callData[plotField]) ?
            (PRICE_FIELDS.includes(selectedPlotType) ? `$${callData[plotField].toFixed(2)}` :
            (Number.isInteger(callData[plotField]) ? callData[plotField].toString() : callData[plotField].toFixed(2))) : 'N/A';
        const putValue = putData && putData[plotField] != null && !isNaN(putData[plotField]) ?
            (PRICE_FIELDS.includes(selectedPlotType) ? `$${putData[plotField].toFixed(2)}` :
            (Number.isInteger(putData[plotField]) ? putData[plotField].toString() : putData[plotField].toFixed(2))) : 'N/A';

        const yMax = Math.max(...calls.map(item => item[plotField] || 0), ...puts.map(item => item[plotField] || 0)) * 1.1;

        Plotly.restyle('options-plot', {
            name: [`Calls: ${callValue}`, `Puts: ${putValue}`]
        }, [0, 1]);

        Plotly.restyle('options-plot', {
            x: [[strike, strike]],
            y: [[0, yMax]],
            name: [`Strike: $${strike.toFixed(2)}`],
            visible: [true]
        }, [3]);
    } catch (error) {
        console.error('Error updating hover line:', error);
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
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
let state = {
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
    totalDates: 0
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
    fetch('/api/get_options_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            symbol: ticker
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || `Failed to fetch data for ${ticker}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log(`Received data for ${ticker}:`, data);
        
        // Update state with the new data
        state.symbol = data.symbol;
        state.currentPrice = data.current_price;
        
        // If we have options data, update the state
        if (data.options_data && data.options_data.length > 0) {
            state.expiryDates = data.expiry_dates;
            state.optionsData = data.options_data;
            
            if (state.currentExpiryIndex >= state.expiryDates.length) {
                state.currentExpiryIndex = 0;
            }
            
            // Update UI with the data we have
            updateExpiryDisplay();
            updatePlot();
        }
        
        state.lastProcessedDates = data.processed_dates;
        state.totalDates = data.total_dates;
        state.lastUpdateTime = new Date();
        
        // Update status based on whether we have partial or complete data
        const timeStr = state.lastUpdateTime.toLocaleTimeString();
        const progressStr = `${state.lastProcessedDates}/${state.totalDates} dates`;
        const progressPct = data.progress ? `${Math.round(data.progress)}%` : '0%';
        
        if (data.status === 'loading' || data.status === 'partial') {
            // Show loading status in the status label only
            elements.statusLabel.textContent = `Loading: ${progressStr} (${progressPct}) | ${timeStr}`;
            
            // Start polling for more data if we're not already polling
            if (!state.isPolling) {
                startPolling(ticker);
            }
        } else {
            // Data is complete
            elements.statusLabel.textContent = `Updated: ${timeStr} | ${state.expiryDates.length} dates loaded`;
            stopPolling();
        }
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        showError(error.message);
        elements.statusLabel.textContent = `Error: ${error.message.substring(0, 30)}...`;
        stopPolling();
    })
    .finally(() => {
        if (!state.isPolling) {
            elements.searchBtn.disabled = false;
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
    
    // Poll every 2 seconds
    state.pollingInterval = setInterval(() => {
        // If we've loaded all dates, stop polling
        if (state.lastProcessedDates >= state.totalDates && state.totalDates > 0) {
            stopPolling();
            return;
        }
        
        fetchOptionsData(ticker, true);
    }, 2000);
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
    console.log("Updating plot");
    if (!state.optionsData || !state.expiryDates || state.expiryDates.length === 0) {
        console.error('Cannot update plot: No data available');
        return;
    }
    
    try {
        const currentDate = state.expiryDates[state.currentExpiryIndex];
        console.log(`Filtering data for date: ${currentDate}`);
        
        // Filter data for current expiry date
        const filteredData = state.optionsData.filter(item => item.expiration === currentDate);
        
        if (!filteredData || filteredData.length === 0) {
            console.warn(`No data available for expiry date ${currentDate}, trying to find another date`);
            
            // Try to find another date with data
            let foundValidDate = false;
            for (let i = 0; i < state.expiryDates.length; i++) {
                const testDate = state.expiryDates[i];
                const testData = state.optionsData.filter(item => item.expiration === testDate);
                if (testData && testData.length > 0) {
                    state.currentExpiryIndex = i;
                    updateExpiryDisplay();
                    // Call updatePlot again with the new date
                    updatePlot();
                    foundValidDate = true;
                    break;
                }
            }
            
            if (!foundValidDate) {
                console.error('No valid data found for any expiry date');
                return;
            }
            return;
        }
        
        console.log(`Found ${filteredData.length} data points for date ${currentDate}`);
        
        // Get selected plot type
        const selectedPlotType = Array.from(elements.plotOptions).find(option => option.checked).value;
        const plotField = FIELD_MAPPING[selectedPlotType];
        
        // Separate calls and puts
        const calls = filteredData.filter(item => item.option_type === 'call');
        const puts = filteredData.filter(item => item.option_type === 'put');
        
        console.log(`Calls: ${calls.length}, Puts: ${puts.length}`);
        
        // Calculate days to expiry
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const expiryDate = new Date(currentDate);
        const dte = Math.max(0, Math.floor((expiryDate - today) / (1000 * 60 * 60 * 24)));
        
        // Filter out null values and create traces for calls and puts
        const callTrace = {
            x: calls.filter(item => item.strike != null && item[plotField] != null)
                 .map(item => item.strike),
            y: calls.filter(item => item.strike != null && item[plotField] != null)
                 .map(item => item[plotField]),
            mode: 'lines',
            name: 'Calls',
            line: { color: 'blue' },
            hoverinfo: 'x+y+name'
        };
        
        const putTrace = {
            x: puts.filter(item => item.strike != null && item[plotField] != null)
                .map(item => item.strike),
            y: puts.filter(item => item.strike != null && item[plotField] != null)
                .map(item => item[plotField]),
            mode: 'lines',
            name: 'Puts',
            line: { color: 'red' },
            hoverinfo: 'x+y+name'
        };
        
        console.log(`Call trace points: ${callTrace.x.length}, Put trace points: ${putTrace.x.length}`);
        
        // Check if we have enough data points to plot
        if (callTrace.x.length < 2 && putTrace.x.length < 2) {
            console.warn(`Not enough data points for ${currentDate} to create a meaningful plot`);
            return;
        }
        
        // Create vertical line for current price
        const yValues = [...callTrace.y.filter(y => y != null && !isNaN(y)), 
                         ...putTrace.y.filter(y => y != null && !isNaN(y))];
        
        // Handle case where we might not have enough data yet
        const maxY = yValues.length > 0 
            ? Math.max(...yValues) * 1.1 
            : 100; // Default if no valid data yet
        
        const priceLine = {
            x: [state.currentPrice, state.currentPrice],
            y: [0, maxY],
            mode: 'lines+text',
            name: `Strike: --`,
            text: [`Price: $${state.currentPrice.toFixed(2)}`, ''],
            textposition: 'top',
            textfont: {
                color: 'black',
                size: 12
            },
            line: {
                color: 'green',
                width: 2,
                dash: 'dash'
            },
            hoverinfo: 'none'
        };
        
        // Create data array for plot
        const data = [callTrace, putTrace, priceLine];
        
        // Create layout
        const layout = {
            title: {
                text: `${state.symbol} Options - ${selectedPlotType} (${dte} DTE)`,
                font: {
                    size: 18
                }
            },
            xaxis: {
                title: 'Strike Price',
                tickprefix: '$'
            },
            yaxis: {
                title: selectedPlotType,
                tickprefix: PRICE_FIELDS.includes(selectedPlotType) ? '$' : ''
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
                t: 50,
                b: 50
            },
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: '#f8f9fa'
        };
        
        console.log("Creating or updating plot");
        
        // Create or update plot
        if (!state.plot) {
            console.log("Creating new plot");
            try {
                // Force a layout calculation before plotting
                const plotDiv = elements.plotContainer;
                console.log(`Plot container dimensions before plot: ${plotDiv.offsetWidth}x${plotDiv.offsetHeight}`);
                
                // Create the plot
                Plotly.newPlot('options-plot', data, layout, {responsive: true})
                    .then(() => {
                        console.log("Plot created successfully");
                        state.plot = document.getElementById('options-plot');
                        
                        // Add event listener for hover using the correct Plotly syntax
                        state.plot.on('plotly_hover', function(data) {
                            if (data.points && data.points.length > 0) {
                                const strike = data.points[0].x;
                                console.log(`Hover detected at strike: ${strike}`);
                                if (strike !== state.hoveredStrike) {
                                    state.hoveredStrike = strike;
                                    updateHoverLine(strike);
                                }
                            }
                        });
                    })
                    .catch(err => {
                        console.error("Error creating plot:", err);
                    });
            } catch (err) {
                console.error("Exception during plot creation:", err);
            }
        } else {
            console.log("Updating existing plot");
            try {
                Plotly.react('options-plot', data, layout)
                    .catch(err => {
                        console.error("Error updating plot:", err);
                    });
            } catch (err) {
                console.error("Exception during plot update:", err);
            }
        }
        
    } catch (error) {
        console.error('Error updating plot:', error);
    }
}

// Helper function to find the value at a specific strike
function findValueAtStrike(data, strike, field) {
    // Filter out null and NaN values first
    const validData = data.filter(item => 
        item.strike != null && 
        item[field] != null && 
        !isNaN(item[field]) && 
        item.strike !== undefined && 
        item[field] !== undefined
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
    console.log(`Updating hover line for strike: ${strike}`);
    if (!state.plot) {
        console.error("Cannot update hover line: Plot not initialized");
        return;
    }
    
    try {
        // Get the plot data from the DOM element
        const plotDiv = document.getElementById('options-plot');
        if (!plotDiv || !plotDiv.data) {
            console.error("Cannot update hover line: Plot data not available");
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
        
        // Update trace names with values
        if (callData && callData[plotField] != null && !isNaN(callData[plotField])) {
            const formattedValue = PRICE_FIELDS.includes(selectedPlotType) ? 
                `$${callData[plotField].toFixed(2)}` : 
                Number.isInteger(callData[plotField]) ? callData[plotField].toString() : callData[plotField].toFixed(2);
            plotData[0].name = `Calls: ${formattedValue}`;
        } else {
            plotData[0].name = 'Calls: N/A';
        }
        
        if (putData && putData[plotField] != null && !isNaN(putData[plotField])) {
            const formattedValue = PRICE_FIELDS.includes(selectedPlotType) ? 
                `$${putData[plotField].toFixed(2)}` : 
                Number.isInteger(putData[plotField]) ? putData[plotField].toString() : putData[plotField].toFixed(2);
            plotData[1].name = `Puts: ${formattedValue}`;
        } else {
            plotData[1].name = 'Puts: N/A';
        }
        
        // Update strike line
        plotData[2].name = `Strike: $${strike.toFixed(2)}`;
        plotData[2].x = [strike, strike];
        
        Plotly.redraw('options-plot')
            .catch(err => {
                console.error("Error redrawing plot:", err);
            });
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
                    autosize: true
                }).catch(err => {
                    console.error("Error resizing plot:", err);
                });
            } catch (err) {
                console.error("Exception during plot resize:", err);
            }
        }
    });
}); 
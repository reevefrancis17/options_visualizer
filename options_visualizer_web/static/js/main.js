// Options Visualizer Web App - Frontend Logic

// Field mapping for plot display (matches the Python backend)
const FIELD_MAPPING = {
    "Spot": "spot",
    "Bid": "bid",
    "Ask": "ask",
    "Volume": "volume",
    "Intrinsic Value": "intrinsic_value",
    "Extrinsic Value": "extrinsic_value"
};

// Price-related fields that need dollar formatting
const PRICE_FIELDS = ["Spot", "Bid", "Ask", "Intrinsic Value", "Extrinsic Value"];

// Global state
let state = {
    symbol: "",
    currentPrice: null,
    expiryDates: [],
    currentExpiryIndex: 0,
    optionsData: [],
    lastUpdateTime: null,
    plot: null
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
    loadingIndicator: document.getElementById('loading-indicator'),
    errorMessage: document.getElementById('error-message')
};

// Initialize the app
function init() {
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
    searchTicker();
}

// Search for a ticker
function searchTicker() {
    const ticker = elements.ticker.value.trim().toUpperCase();
    if (!ticker || !ticker.match(/^[A-Z0-9]+$/)) {
        showError('Invalid ticker symbol. Please enter a valid stock symbol.');
        return;
    }
    
    // Update UI state
    elements.searchBtn.disabled = true;
    showLoading(true);
    hideError();
    elements.statusLabel.textContent = `Loading ${ticker} data...`;
    
    // Fetch data from the API
    fetch('/api/get_options_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbol: ticker })
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
        // Update state with the new data
        state.symbol = data.symbol;
        state.currentPrice = data.current_price;
        state.expiryDates = data.expiry_dates;
        state.optionsData = data.options_data;
        state.currentExpiryIndex = 0;
        state.lastUpdateTime = new Date();
        
        // Update UI
        updateExpiryDisplay();
        updatePlot();
        
        // Update status
        const timeStr = state.lastUpdateTime.toLocaleTimeString();
        elements.statusLabel.textContent = `Updated: ${timeStr} | ${state.expiryDates.length} dates loaded`;
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        showError(error.message);
        elements.statusLabel.textContent = `Error: ${error.message.substring(0, 30)}...`;
    })
    .finally(() => {
        elements.searchBtn.disabled = false;
        showLoading(false);
    });
}

// Update the expiry date display and navigation buttons
function updateExpiryDisplay() {
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
    if (!state.optionsData || !state.expiryDates || state.expiryDates.length === 0) {
        console.error('Cannot update plot: No data available');
        return;
    }
    
    try {
        const currentDate = state.expiryDates[state.currentExpiryIndex];
        
        // Filter data for current expiry date
        const filteredData = state.optionsData.filter(item => item.expiration === currentDate);
        
        if (!filteredData || filteredData.length === 0) {
            console.error(`No data available for expiry date ${currentDate}`);
            return;
        }
        
        // Get selected plot type
        const selectedPlotType = Array.from(elements.plotOptions).find(option => option.checked).value;
        const plotField = FIELD_MAPPING[selectedPlotType];
        
        // Separate calls and puts
        const calls = filteredData.filter(item => item.option_type === 'call');
        const puts = filteredData.filter(item => item.option_type === 'put');
        
        // Calculate days to expiry
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const expiryDate = new Date(currentDate);
        const dte = Math.max(0, Math.floor((expiryDate - today) / (1000 * 60 * 60 * 24)));
        
        // Filter out null values and create traces for calls and puts
        const callTrace = {
            x: calls.filter(item => item.strike !== null && item[plotField] !== null)
                 .map(item => item.strike),
            y: calls.filter(item => item.strike !== null && item[plotField] !== null)
                 .map(item => item[plotField]),
            mode: 'lines',
            name: 'Calls',
            line: { color: 'blue' },
            hoverinfo: 'x+y+name'
        };
        
        const putTrace = {
            x: puts.filter(item => item.strike !== null && item[plotField] !== null)
                .map(item => item.strike),
            y: puts.filter(item => item.strike !== null && item[plotField] !== null)
                .map(item => item[plotField]),
            mode: 'lines',
            name: 'Puts',
            line: { color: 'red' },
            hoverinfo: 'x+y+name'
        };
        
        // Create vertical line for current price
        const maxY = Math.max(
            ...callTrace.y.filter(y => y !== null && !isNaN(y)),
            ...putTrace.y.filter(y => y !== null && !isNaN(y))
        ) * 1.1;
        
        const priceLine = {
            x: [state.currentPrice, state.currentPrice],
            y: [0, maxY],
            mode: 'lines+text',
            name: `Spot: $${state.currentPrice.toFixed(2)}`,
            text: [`Spot: $${state.currentPrice.toFixed(2)}`, ''],
            textposition: 'top',
            textfont: {
                color: 'green',
                size: 12
            },
            line: { color: 'green', dash: 'dash' },
            hoverinfo: 'none'
        };
        
        // Set up layout
        const layout = {
            title: `${selectedPlotType} vs Strike Price - ${currentDate} (DTE: ${dte})`,
            xaxis: {
                title: 'Strike Price ($)',
                tickprefix: '$'
            },
            yaxis: {
                title: PRICE_FIELDS.includes(selectedPlotType) ? `${selectedPlotType} ($)` : selectedPlotType,
                tickprefix: PRICE_FIELDS.includes(selectedPlotType) ? '$' : ''
            },
            hovermode: 'closest',
            grid: { rows: 1, columns: 1, pattern: 'independent' },
            margin: { l: 50, r: 20, t: 50, b: 50 },
            autosize: true
        };
        
        // Configure options to hide modebar and enable comparison
        const config = {
            displayModeBar: false,
            responsive: true,
            scrollZoom: true,
            showTips: false,
            compareTraces: true
        };
        
        // Create the plot
        Plotly.newPlot(elements.plotContainer, [callTrace, putTrace, priceLine], layout, config);
        
        // Add hover event for crosshair-like functionality
        elements.plotContainer.on('plotly_hover', function(data) {
            const pointData = data.points[0];
            const curveNumber = pointData.curveNumber;
            const pointIndex = pointData.pointIndex;
            
            // Update the trace names with values
            const traces = data.points[0].fullData;
            if (curveNumber === 0 && pointIndex < callTrace.y.length) { // Calls
                const value = callTrace.y[pointIndex];
                if (value !== null && !isNaN(value)) {
                    traces[0].name = `Calls: ${PRICE_FIELDS.includes(selectedPlotType) ? '$' : ''}${value.toFixed(2)}`;
                    Plotly.redraw(elements.plotContainer);
                }
            } else if (curveNumber === 1 && pointIndex < putTrace.y.length) { // Puts
                const value = putTrace.y[pointIndex];
                if (value !== null && !isNaN(value)) {
                    traces[1].name = `Puts: ${PRICE_FIELDS.includes(selectedPlotType) ? '$' : ''}${value.toFixed(2)}`;
                    Plotly.redraw(elements.plotContainer);
                }
            }
        });
        
        // Reset trace names on mouseout
        elements.plotContainer.on('plotly_unhover', function() {
            const traces = elements.plotContainer.data;
            traces[0].name = 'Calls';
            traces[1].name = 'Puts';
            Plotly.redraw(elements.plotContainer);
        });
        
    } catch (error) {
        console.error('Error updating plot:', error);
        showError(`Error updating plot: ${error.message}`);
    }
}

// Show/hide loading indicator
function showLoading(show) {
    elements.loadingIndicator.classList.toggle('hidden', !show);
}

// Show error message
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.classList.remove('hidden');
}

// Hide error message
function hideError() {
    elements.errorMessage.classList.add('hidden');
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    init();
    
    // Add window resize handler to make plot responsive
    window.addEventListener('resize', function() {
        if (state.optionsData && state.optionsData.length > 0) {
            Plotly.relayout(elements.plotContainer, {
                autosize: true
            });
        }
    });
}); 
// Global variable to store the current data
let currentOptionsData = null;
let currentSortColumn = 'strike';
let currentSortDirection = 'asc';

// Add event listener for keyboard events
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('symbolInput');
    input.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            fetchData();
        }
    });
});

function updateStatus(message) {
    const status = document.getElementById('statusMessage');
    status.textContent = message;
}

async function fetchData() {
    const symbol = document.getElementById('symbolInput').value.toUpperCase();
    
    if (!symbol) {
        updateStatus('Please enter a symbol');
        return;
    }
    
    updateStatus('Fetching data...');
    
    try {
        const response = await fetch(`http://localhost:5001/api/options/${symbol}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            updateStatus('Error: ' + data.error);
            return;
        }
        
        if (!data.options || !data.options.length) {
            updateStatus('No option data available for ' + symbol);
            return;
        }
        
        // Store the data globally
        currentOptionsData = data;
        // Create table with default sort
        createTable(data);
        updateStatus(`Showing ${data.options.length} options for ${symbol}`);
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error: ' + error.message);
    }
}

function sortData(column) {
    if (!currentOptionsData) return;
    
    // Toggle sort direction if clicking the same column
    if (column === currentSortColumn) {
        currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        currentSortColumn = column;
        currentSortDirection = 'asc';
    }
    
    // Sort the data
    currentOptionsData.options.sort((a, b) => {
        const valueA = a[column];
        const valueB = b[column];
        const multiplier = currentSortDirection === 'asc' ? 1 : -1;
        return (valueA - valueB) * multiplier;
    });
    
    // Recreate the table with sorted data
    createTable(currentOptionsData);
}

function createSortHeader(column, displayName) {
    const isSorted = currentSortColumn === column;
    const direction = isSorted ? currentSortDirection : '';
    const sortIconClass = direction ? `sort-icon ${direction}` : 'sort-icon';
    
    return `
        <th onclick="sortData('${column}')">
            ${displayName}
            <span class="${sortIconClass}"></span>
        </th>
    `;
}

function createTable(data) {
    const container = document.getElementById('optionsTable');
    
    // Create header
    const header = `<h2>Options Expiring ${data.friday_date}</h2>`;
    
    // Create table with sortable headers
    let tableHtml = `
        <table>
            <tr>
                ${createSortHeader('strike', 'Strike Price')}
                ${createSortHeader('call_price', 'Call Price')}
                ${createSortHeader('put_price', 'Put Price')}
            </tr>
    `;
    
    // Add rows
    data.options.forEach(option => {
        tableHtml += `
            <tr>
                <td>$${option.strike.toFixed(2)}</td>
                <td>$${option.call_price.toFixed(2)}</td>
                <td>$${option.put_price.toFixed(2)}</td>
            </tr>
        `;
    });
    
    tableHtml += '</table>';
    
    // Set the HTML
    container.innerHTML = header + tableHtml;
} 
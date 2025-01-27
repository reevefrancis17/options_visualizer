// Initialize the options table
const optionsTable = new OptionsTable('optionsTable');

// Add event listener for keyboard events
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('symbolInput');
    input.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            fetchData();
        }
    });
});

async function fetchData() {
    const symbol = document.getElementById('symbolInput').value.toUpperCase();
    
    if (!symbol) {
        document.getElementById('statusMessage').textContent = 'Please enter a symbol';
        return;
    }
    
    document.getElementById('statusMessage').textContent = 'Fetching data...';
    
    try {
        await optionsTable.initialize(symbol);
        document.getElementById('statusMessage').textContent = `Showing options for ${symbol}`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('statusMessage').textContent = 'Error: ' + error.message;
    }
} 
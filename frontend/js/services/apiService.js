class ApiService {
    constructor(baseUrl = 'http://localhost:5001') {
        this.baseUrl = baseUrl;
    }

    async fetchOptionChain(symbol, date = null) {
        try {
            const url = date 
                ? `${this.baseUrl}/api/options/${symbol}?date=${date}`
                : `${this.baseUrl}/api/options/${symbol}`;
                
            console.log(`Fetching option chain for ${symbol}${date ? ` expiring ${date}` : ''}`);
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            if (!data.friday_date && date) {
                data.friday_date = date;
            }
            return data;
        } catch (error) {
            // Log to server
            await this.logError('fetchOptionChain', error.message, {symbol, date});
            throw error;
        }
    }

    async fetchExpiryDates(symbol) {
        try {
            console.log(`Fetching expiry dates for ${symbol}`);
            const response = await fetch(`${this.baseUrl}/api/expiry_dates/${symbol}`);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch expiry dates: ${errorText}`);
            }
            
            const data = await response.json();
            return data.dates;
        } catch (error) {
            await this.logError('fetchExpiryDates', error.message, {symbol});
            throw error;
        }
    }

    async logError(method, errorMessage, context = {}) {
        try {
            const logData = {
                timestamp: new Date().toISOString(),
                method,
                error: errorMessage,
                context
            };

            await fetch(`${this.baseUrl}/api/log`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(logData)
            });
        } catch (e) {
            console.error('Failed to log error:', e);
        }
    }
} 
class ApiService {
    constructor(baseUrl = 'http://localhost:5001') {
        this.baseUrl = baseUrl;
    }

    async fetchOptionChain(symbol) {
        try {
            const response = await fetch(`${this.baseUrl}/api/options/${symbol}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }
} 
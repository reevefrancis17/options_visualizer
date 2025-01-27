class OptionsTable {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentSortColumn = 'strike';
        this.currentSortDirection = 'asc';
        this.currentSymbol = null;
        this.expiryDates = [];
        this.currentExpiryIndex = 0;
        this.api = new ApiService();
    }

    async initialize(symbol) {
        this.currentSymbol = symbol;
        try {
            // Fetch all available expiry dates for this symbol
            const dates = await this.api.fetchExpiryDates(symbol);
            if (!dates || dates.length === 0) {
                throw new Error(`No expiry dates available for ${symbol}`);
            }
            
            this.expiryDates = dates.sort();
            const thisFriday = this.getThisFriday();
            console.log('This Friday:', thisFriday);
            console.log('Available dates:', this.expiryDates);
            
            this.currentExpiryIndex = this.expiryDates.findIndex(date => 
                date >= thisFriday);
            if (this.currentExpiryIndex === -1) {
                this.currentExpiryIndex = 0;
            }
            
            console.log('Selected expiry index:', this.currentExpiryIndex);
            await this.fetchAndDisplayData();
        } catch (error) {
            await this.api.logError('initialize', error.message, {symbol});
            throw error;
        }
    }

    getThisFriday() {
        const today = new Date();
        const friday = new Date(today);
        friday.setDate(today.getDate() + (5 - today.getDay()) % 7);
        return friday.toISOString().split('T')[0];
    }

    async fetchAndDisplayData() {
        try {
            const date = this.expiryDates[this.currentExpiryIndex];
            console.log('Fetching data for date:', date);
            if (!date) {
                throw new Error('No expiry date selected');
            }
            
            const data = await this.api.fetchOptionChain(this.currentSymbol, date);
            if (!data.friday_date) {
                data.friday_date = date; // Ensure we have a date to display
            }
            this.render(data);
        } catch (error) {
            await this.api.logError('fetchAndDisplayData', error.message, {
                symbol: this.currentSymbol,
                date: this.expiryDates[this.currentExpiryIndex]
            });
            throw error;
        }
    }

    async handleNavigation(direction) {
        const newIndex = this.currentExpiryIndex + (direction === 'next' ? 1 : -1);
        if (newIndex >= 0 && newIndex < this.expiryDates.length) {
            this.currentExpiryIndex = newIndex;
            await this.fetchAndDisplayData();
        }
    }

    render(data) {
        if (!data?.options) return;

        this.currentData = data;  // Store the current data
        const headerHtml = this.createHeader(data.friday_date);
        const tableHtml = this.createTable(data.options);
        this.container.innerHTML = headerHtml + tableHtml;

        // Add event listeners to navigation buttons after rendering
        this.setupNavigationButtons();
    }

    setupNavigationButtons() {
        const prevBtn = this.container.querySelector('.nav-button.prev');
        const nextBtn = this.container.querySelector('.nav-button.next');
        
        if (prevBtn && nextBtn) {
            prevBtn.disabled = this.currentExpiryIndex === 0;
            nextBtn.disabled = this.currentExpiryIndex === this.expiryDates.length - 1;

            // Remove old event listeners if any
            prevBtn.replaceWith(prevBtn.cloneNode(true));
            nextBtn.replaceWith(nextBtn.cloneNode(true));

            // Add new event listeners
            this.container.querySelector('.nav-button.prev')
                .addEventListener('click', () => this.handleNavigation('prev'));
            this.container.querySelector('.nav-button.next')
                .addEventListener('click', () => this.handleNavigation('next'));
            
            // Add visual feedback
            console.log('Navigation buttons set up:', {
                prevDisabled: prevBtn.disabled,
                nextDisabled: nextBtn.disabled,
                currentIndex: this.currentExpiryIndex,
                totalDates: this.expiryDates.length
            });
        }
    }

    createHeader(expiryDate) {
        const spotPrice = this.currentData?.spot_price;
        return `
            <div style="margin-bottom: 20px;">
                <div style="margin-bottom: 10px;">
                    <h2 style="margin: 0; font-size: 1.5em;">
                        ${this.currentSymbol?.toUpperCase() || ''} 
                        ${spotPrice ? `$${spotPrice.toFixed(2)}` : ''}
                    </h2>
                </div>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="expiry-nav">
                        <button 
                            class="nav-button prev" 
                            ${this.currentExpiryIndex === 0 ? 'disabled' : ''}
                            title="Previous expiry date"
                        ></button>
                        <span style="margin: 0 10px; font-size: 1.2em;">
                            ${expiryDate || 'Loading...'}
                        </span>
                        <button 
                            class="nav-button next" 
                            ${this.currentExpiryIndex === this.expiryDates.length - 1 ? 'disabled' : ''}
                            title="Next expiry date"
                        ></button>
                    </div>
                </div>
            </div>
        `;
    }

    createTable(options) {
        if (!options || options.length === 0) return '';

        return `
            <table>
                <thead>
                    <tr>
                        <th data-sort="strike" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('strike')">
                            Strike ${this.getSortIcon('strike')}
                        </th>
                        <th colspan="4" style="text-align: center;">Calls</th>
                        <th colspan="4" style="text-align: center;">Puts</th>
                    </tr>
                    <tr>
                        <th></th>
                        <th data-sort="call_price" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('call_price')">
                            Last ${this.getSortIcon('call_price')}
                        </th>
                        <th data-sort="call_bid" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('call_bid')">
                            Bid ${this.getSortIcon('call_bid')}
                        </th>
                        <th data-sort="call_ask" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('call_ask')">
                            Ask ${this.getSortIcon('call_ask')}
                        </th>
                        <th data-sort="call_volume" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('call_volume')">
                            Vol ${this.getSortIcon('call_volume')}
                        </th>
                        <th data-sort="put_price" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('put_price')">
                            Last ${this.getSortIcon('put_price')}
                        </th>
                        <th data-sort="put_bid" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('put_bid')">
                            Bid ${this.getSortIcon('put_bid')}
                        </th>
                        <th data-sort="put_ask" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('put_ask')">
                            Ask ${this.getSortIcon('put_ask')}
                        </th>
                        <th data-sort="put_volume" style="text-align: center; cursor: pointer;" onclick="optionsTable.handleSort('put_volume')">
                            Vol ${this.getSortIcon('put_volume')}
                        </th>
                    </tr>
                </thead>
                <tbody>
                    ${options.map(option => `
                        <tr>
                            <td style="text-align: center;">$${option.strike.toFixed(2)}</td>
                            <td style="text-align: center;">$${option.call_price.toFixed(2)}</td>
                            <td style="text-align: center;">$${option.call_bid.toFixed(2)}</td>
                            <td style="text-align: center;">$${option.call_ask.toFixed(2)}</td>
                            <td style="text-align: center;">${option.call_volume.toLocaleString()}</td>
                            <td style="text-align: center;">$${option.put_price.toFixed(2)}</td>
                            <td style="text-align: center;">$${option.put_bid.toFixed(2)}</td>
                            <td style="text-align: center;">$${option.put_ask.toFixed(2)}</td>
                            <td style="text-align: center;">${option.put_volume.toLocaleString()}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    getSortIcon(column) {
        if (this.currentSortColumn !== column) {
            return '<span class="sort-icon"></span>';
        }
        return `<span class="sort-icon ${this.currentSortDirection}"></span>`;
    }

    handleSort(column) {
        if (!this.currentData?.options) return;

        // Toggle sort direction if clicking the same column
        if (column === this.currentSortColumn) {
            this.currentSortDirection = this.currentSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.currentSortColumn = column;
            this.currentSortDirection = 'asc';
        }

        // Sort the data
        this.currentData.options.sort((a, b) => {
            const valueA = a[column];
            const valueB = b[column];
            const multiplier = this.currentSortDirection === 'asc' ? 1 : -1;
            return (valueA - valueB) * multiplier;
        });

        // Re-render the table
        this.render(this.currentData);
    }
}
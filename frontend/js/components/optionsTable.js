class OptionsTable {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentSortColumn = 'strike';
        this.currentSortDirection = 'asc';
    }

    render(data) {
        if (!data?.options) return;

        const header = `<h2>Options Expiring ${data.friday_date}</h2>`;
        const table = this.createTable(data.options);
        this.container.innerHTML = header + table;
    }

    createTable(options) {
        return `
            <table>
                <tr>
                    ${this.createSortHeader('strike', 'Strike')}
                    ${this.createSortHeader('call_price', 'Calls')}
                    ${this.createSortHeader('put_price', 'Puts')}
                </tr>
                ${this.createRows(options)}
            </table>
        `;
    }

    createSortHeader(column, displayName) {
        const isSorted = this.currentSortColumn === column;
        const direction = isSorted ? this.currentSortDirection : '';
        const sortIconClass = `sort-icon ${direction}`;
        
        return `
            <th onclick="optionsTable.handleSort('${column}')">
                ${displayName}
                <span class="${sortIconClass}"></span>
            </th>
        `;
    }

    createRows(options) {
        return options.map(option => `
            <tr>
                <td>$${option.strike.toFixed(2)}</td>
                <td>$${option.call_price.toFixed(2)}</td>
                <td>$${option.put_price.toFixed(2)}</td>
            </tr>
        `).join('');
    }

    handleSort(column) {
        if (column === this.currentSortColumn) {
            this.currentSortDirection = this.currentSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.currentSortColumn = column;
            this.currentSortDirection = 'asc';
        }
        
        const sortedData = dataService.sortData(column, this.currentSortDirection);
        this.render(sortedData);
    }
}

// Initialize the table
const optionsTable = new OptionsTable('optionsTable'); 
class DataService {
    constructor() {
        this.currentData = null;
    }

    setData(data) {
        this.currentData = data;
    }

    getData() {
        return this.currentData;
    }

    sortData(column, direction) {
        if (!this.currentData?.options) return;

        this.currentData.options.sort((a, b) => {
            const valueA = a[column];
            const valueB = b[column];
            const multiplier = direction === 'asc' ? 1 : -1;
            return (valueA - valueB) * multiplier;
        });

        return this.currentData;
    }
} 
// Initialize services
const api = new ApiService();
const dataService = new DataService();
const statusMessage = new StatusMessage('statusMessage');

// Initialize views
const tableView = new TableView('table-view');
const surfaceView = new SurfaceView('surface-view');
const greeksView = new GreeksView('greeks-view');

// Handle navigation
document.querySelectorAll('.main-nav a').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const pageId = e.target.dataset.page;
        switchPage(pageId);
    });
});

function switchPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show selected page
    document.getElementById(pageId).classList.add('active');
    
    // Update nav
    document.querySelectorAll('.main-nav a').forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageId);
    });
} 
# Options Visualizer Web App

A web-based version of the Options Visualizer that allows you to visualize options data for any stock symbol. This application consists of a Flask backend that fetches and processes options data, and a web frontend that displays the data in interactive charts.

## Features

- Search for any stock symbol to view its options data
- Navigate between different expiration dates
- View different metrics (Spot, Bid, Ask, Volume, Intrinsic Value, Extrinsic Value)
- Interactive charts with hover functionality
- Responsive design that works on desktop and mobile

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/options_visualizer.git
   cd options_visualizer
   ```

2. Install the required dependencies:
   ```
   pip install -r options_visualizer_web/requirements.txt
   ```

### Running the Application

1. Start the Flask server:
   ```
   cd options_visualizer_web
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5001
   ```

## Hosting on Your Local Machine

To make the application accessible from anywhere:

1. **Port Forwarding**: Configure your router to forward port 80 (external) to port 5001 (internal) on your machine's local IP.

2. **Dynamic DNS**: Register with a dynamic DNS service (e.g., No-IP) to get a domain that points to your home IP address, which will update automatically if your IP changes.

3. **Domain Setup**: If you have your own domain, configure it to point to your dynamic DNS address.

4. **Firewall Configuration**: Ensure your firewall allows incoming traffic on port 80.

5. **Running the Server**: Keep your machine on and the server running to maintain access.

## Security Considerations

- This setup uses HTTP by default. For production use, consider adding HTTPS with Let's Encrypt and Nginx as a reverse proxy.
- Be aware that exposing your local machine to the internet comes with security risks. Consider implementing additional security measures.

## Troubleshooting

- If you encounter issues with data fetching, check your internet connection and ensure you're not hitting Yahoo Finance's rate limits.
- If the application is not accessible externally, verify your port forwarding and firewall settings.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
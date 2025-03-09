import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the Flask app
try:
    from options_visualizer_web.app import app
except ImportError:
    # If not found, skip the tests
    app = None


@unittest.skipIf(app is None, "Frontend app not found")
class TestFrontendApp(unittest.TestCase):
    """Test cases for the frontend Flask application."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_page(self):
        """Test that the home page loads correctly."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)
        self.assertIn(b'Options Chain Visualizer', response.data)
    
    def test_health_check(self):
        """Test that the health check endpoint returns OK."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
    
    @patch('requests.get')
    def test_backend_proxy(self, mock_get):
        """Test that the backend proxy endpoint works correctly."""
        # Mock the backend response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'success',
            'data': {'ticker': 'AAPL', 'price': 150.0}
        }
        mock_get.return_value = mock_response
        
        # Test the proxy endpoint
        response = self.app.get('/api/proxy/options/AAPL')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['ticker'], 'AAPL')


if __name__ == '__main__':
    unittest.main() 
"""
End-to-end tests for the Options Visualizer web application.

These tests require:
1. Both the frontend and backend servers to be running
2. Selenium WebDriver to be installed
3. Chrome or Firefox browser to be installed

To run these tests:
1. Start the backend server: python -m options_visualizer_backend.app
2. Start the frontend server: python -m options_visualizer_web.app
3. Run the tests: pytest tests/test_e2e.py
"""
import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select


# Skip these tests if Selenium is not installed
pytest.importorskip("selenium")


@pytest.fixture(scope="module")
def driver():
    """Set up and tear down the WebDriver."""
    # Skip the test if running in CI environment without browser
    try:
        # Try to create a Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
    except:
        try:
            # Try Firefox if Chrome fails
            options = webdriver.FirefoxOptions()
            options.add_argument("--headless")
            driver = webdriver.Firefox(options=options)
        except:
            pytest.skip("WebDriver could not be initialized")
    
    driver.implicitly_wait(10)
    yield driver
    driver.quit()


@pytest.mark.skip(reason="Requires running frontend server")
@pytest.mark.e2e
def test_load_homepage(driver):
    """Test that the homepage loads correctly."""
    # Navigate to the homepage
    driver.get("http://localhost:5001")
    
    # Check that the page title is correct
    assert "Options Visualizer" in driver.title
    
    # Check that the search form is present
    search_form = driver.find_element(By.ID, "search-form")
    assert search_form is not None
    
    # Check that the ticker input is present
    ticker_input = driver.find_element(By.ID, "ticker-input")
    assert ticker_input is not None


@pytest.mark.skip(reason="Requires running frontend server")
@pytest.mark.e2e
def test_search_for_ticker(driver):
    """Test searching for a ticker symbol."""
    # Navigate to the homepage
    driver.get("http://localhost:5001")
    
    # Find the ticker input and search button
    ticker_input = driver.find_element(By.ID, "ticker-input")
    search_button = driver.find_element(By.ID, "search-button")
    
    # Enter a ticker symbol and submit the form
    ticker_input.send_keys("SPY")
    search_button.click()
    
    # Wait for the data to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "options-data"))
    )
    
    # Check that the options data is displayed
    options_data = driver.find_element(By.ID, "options-data")
    assert options_data is not None
    
    # Check that the ticker symbol is displayed
    ticker_display = driver.find_element(By.ID, "ticker-display")
    assert "SPY" in ticker_display.text


@pytest.mark.skip(reason="Requires running frontend server")
@pytest.mark.e2e
def test_toggle_metrics(driver):
    """Test toggling between different metrics."""
    # Navigate to the homepage and search for a ticker
    driver.get("http://localhost:5001")
    
    # Find the ticker input and search button
    ticker_input = driver.find_element(By.ID, "ticker-input")
    search_button = driver.find_element(By.ID, "search-button")
    
    # Enter a ticker symbol and submit the form
    ticker_input.send_keys("SPY")
    search_button.click()
    
    # Wait for the data to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "options-data"))
    )
    
    # Find the metric toggle buttons
    last_price_button = driver.find_element(By.ID, "last-price-button")
    implied_vol_button = driver.find_element(By.ID, "implied-vol-button")
    
    # Click the implied volatility button
    implied_vol_button.click()
    
    # Wait for the plot to update
    time.sleep(1)
    
    # Check that the implied volatility data is displayed
    plot = driver.find_element(By.ID, "options-plot")
    assert plot is not None
    
    # Click the last price button
    last_price_button.click()
    
    # Wait for the plot to update
    time.sleep(1)
    
    # Check that the last price data is displayed
    plot = driver.find_element(By.ID, "options-plot")
    assert plot is not None


@pytest.mark.skip(reason="Requires running frontend server")
@pytest.mark.e2e
def test_expiry_navigation(driver):
    """Test navigating between different expiration dates."""
    # Navigate to the homepage and search for a ticker
    driver.get("http://localhost:5001")
    
    # Find the ticker input and search button
    ticker_input = driver.find_element(By.ID, "ticker-input")
    search_button = driver.find_element(By.ID, "search-button")
    
    # Enter a ticker symbol and submit the form
    ticker_input.send_keys("SPY")
    search_button.click()
    
    # Wait for the data to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "options-data"))
    )
    
    # Find the expiry selector
    expiry_selector = driver.find_element(By.ID, "expiry-selector")
    
    # Get the available expiry dates
    expiry_options = Select(expiry_selector).options
    
    # Make sure there are at least 2 expiry dates
    assert len(expiry_options) >= 2
    
    # Select the second expiry date
    Select(expiry_selector).select_by_index(1)
    
    # Wait for the data to update
    time.sleep(1)
    
    # Check that the options data is still displayed
    options_data = driver.find_element(By.ID, "options-data")
    assert options_data is not None 
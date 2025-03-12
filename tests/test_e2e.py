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


@pytest.mark.e2e
def test_load_homepage(driver):
    """Test that the homepage loads correctly."""
    # Navigate to the homepage
    driver.get("http://localhost:5001")
    
    # Check that the page title is correct
    assert "Options Visualizer" in driver.title
    
    # Check that the search form is present
    search_input = driver.find_element(By.ID, "ticker-input")
    assert search_input is not None
    
    # Check that the search button is present
    search_button = driver.find_element(By.ID, "search-button")
    assert search_button is not None


@pytest.mark.e2e
def test_search_for_ticker(driver):
    """Test searching for a ticker symbol."""
    # Navigate to the homepage
    driver.get("http://localhost:5001")
    
    # Enter a ticker symbol
    search_input = driver.find_element(By.ID, "ticker-input")
    search_input.clear()
    search_input.send_keys("SPY")
    
    # Click the search button
    search_button = driver.find_element(By.ID, "search-button")
    search_button.click()
    
    # Wait for the options data to load
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "options-plot"))
        )
    except TimeoutException:
        pytest.skip("Options data did not load in time")
    
    # Check that the plot is displayed
    plot = driver.find_element(By.ID, "options-plot")
    assert plot is not None
    
    # Check that the ticker symbol is displayed
    assert "SPY" in driver.page_source


@pytest.mark.e2e
def test_toggle_metrics(driver):
    """Test toggling between different metrics."""
    # Navigate to the homepage and search for a ticker
    driver.get("http://localhost:5001")
    search_input = driver.find_element(By.ID, "ticker-input")
    search_input.clear()
    search_input.send_keys("SPY")
    search_button = driver.find_element(By.ID, "search-button")
    search_button.click()
    
    # Wait for the options data to load
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "options-plot"))
        )
    except TimeoutException:
        pytest.skip("Options data did not load in time")
    
    # Find the metric radio buttons
    metric_radios = driver.find_elements(By.NAME, "metric")
    
    # Check that we have multiple metric options
    assert len(metric_radios) > 1
    
    # Toggle to a different metric (e.g., Delta)
    for radio in metric_radios:
        if radio.get_attribute("value") == "delta":
            radio.click()
            break
    
    # Wait for the plot to update
    time.sleep(2)
    
    # Check that the plot is still displayed
    plot = driver.find_element(By.ID, "options-plot")
    assert plot is not None
    
    # Check that the metric name is displayed
    assert "Delta" in driver.page_source


@pytest.mark.e2e
def test_expiry_navigation(driver):
    """Test navigating between different expiration dates."""
    # Navigate to the homepage and search for a ticker
    driver.get("http://localhost:5001")
    search_input = driver.find_element(By.ID, "ticker-input")
    search_input.clear()
    search_input.send_keys("SPY")
    search_button = driver.find_element(By.ID, "search-button")
    search_button.click()
    
    # Wait for the options data to load
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "options-plot"))
        )
    except TimeoutException:
        pytest.skip("Options data did not load in time")
    
    # Find the expiry navigation buttons
    next_expiry_button = driver.find_element(By.ID, "next-expiry")
    
    # Check that the button exists
    assert next_expiry_button is not None
    
    # Get the current expiry date
    current_expiry = driver.find_element(By.ID, "current-expiry").text
    
    # Click the next expiry button
    next_expiry_button.click()
    
    # Wait for the plot to update
    time.sleep(2)
    
    # Get the new expiry date
    new_expiry = driver.find_element(By.ID, "current-expiry").text
    
    # Check that the expiry date has changed
    assert new_expiry != current_expiry 
import pytest
import numpy as np
from python.models.black_scholes import (
    d1, d2, call_price, put_price, delta, gamma, theta, vega, rho,
    implied_volatility, calculate_all_greeks, calculate_iv_surface
)

def test_d1():
    """Test the d1 function."""
    # Normal case
    result = d1(100, 100, 0.05, 0.2, 1)
    assert isinstance(result, float)
    assert result == pytest.approx(0.05, abs=0.01)
    
    # Edge cases
    assert d1(100, 0, 0.05, 0.2, 1) == float('inf')
    assert d1(0, 100, 0.05, 0.2, 1) == float('-inf')
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    result = d1(s, k, 0.05, 0.2, 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

def test_d2():
    """Test the d2 function."""
    # Normal case
    result = d2(100, 100, 0.05, 0.2, 1)
    assert isinstance(result, float)
    assert result == pytest.approx(-0.15, abs=0.01)
    
    # Edge cases
    assert d2(100, 0, 0.05, 0.2, 1) == float('inf')
    assert d2(0, 100, 0.05, 0.2, 1) == float('-inf')
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    result = d2(s, k, 0.05, 0.2, 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

def test_call_price():
    """Test the call_price function."""
    # At the money
    price = call_price(100, 100, 0.05, 0.2, 1)
    assert isinstance(price, float)
    assert price == pytest.approx(10.45, abs=0.1)
    
    # In the money
    price = call_price(110, 100, 0.05, 0.2, 1)
    assert price > 10
    
    # Out of the money
    price = call_price(90, 100, 0.05, 0.2, 1)
    assert price < 10
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    prices = call_price(s, k, 0.05, 0.2, 1)
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (3,)
    assert np.all(prices >= 0)

def test_put_price():
    """Test the put_price function."""
    # At the money
    price = put_price(100, 100, 0.05, 0.2, 1)
    assert isinstance(price, float)
    assert price == pytest.approx(5.57, abs=0.1)
    
    # In the money
    price = put_price(90, 100, 0.05, 0.2, 1)
    assert price > 5
    
    # Out of the money
    price = put_price(110, 100, 0.05, 0.2, 1)
    assert price < 5
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    prices = put_price(s, k, 0.05, 0.2, 1)
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (3,)
    assert np.all(prices >= 0)

def test_delta():
    """Test the delta function."""
    # Call delta
    call_delta = delta(100, 100, 0.05, 0.2, 1, option_type='call')
    assert isinstance(call_delta, float)
    assert 0.5 <= call_delta <= 0.7
    
    # Put delta
    put_delta = delta(100, 100, 0.05, 0.2, 1, option_type='put')
    assert isinstance(put_delta, float)
    assert -0.5 >= put_delta >= -0.7
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    call_deltas = delta(s, k, 0.05, 0.2, 1, option_type='call')
    assert isinstance(call_deltas, np.ndarray)
    assert call_deltas.shape == (3,)
    assert np.all(call_deltas >= 0) and np.all(call_deltas <= 1)

def test_gamma():
    """Test the gamma function."""
    # Gamma is the same for calls and puts
    call_gamma = gamma(100, 100, 0.05, 0.2, 1)
    assert isinstance(call_gamma, float)
    assert call_gamma > 0
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    gammas = gamma(s, k, 0.05, 0.2, 1)
    assert isinstance(gammas, np.ndarray)
    assert gammas.shape == (3,)
    assert np.all(gammas >= 0)

def test_theta():
    """Test the theta function."""
    # Call theta (should be negative)
    call_theta = theta(100, 100, 0.05, 0.2, 1, option_type='call')
    assert isinstance(call_theta, float)
    assert call_theta < 0
    
    # Put theta (should be negative)
    put_theta = theta(100, 100, 0.05, 0.2, 1, option_type='put')
    assert isinstance(put_theta, float)
    assert put_theta < 0
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    call_thetas = theta(s, k, 0.05, 0.2, 1, option_type='call')
    assert isinstance(call_thetas, np.ndarray)
    assert call_thetas.shape == (3,)
    assert np.all(call_thetas <= 0)

def test_vega():
    """Test the vega function."""
    # Vega is the same for calls and puts
    call_vega = vega(100, 100, 0.05, 0.2, 1)
    assert isinstance(call_vega, float)
    assert call_vega > 0
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    vegas = vega(s, k, 0.05, 0.2, 1)
    assert isinstance(vegas, np.ndarray)
    assert vegas.shape == (3,)
    assert np.all(vegas >= 0)

def test_rho():
    """Test the rho function."""
    # Call rho (should be positive)
    call_rho = rho(100, 100, 0.05, 0.2, 1, option_type='call')
    assert isinstance(call_rho, float)
    assert call_rho > 0
    
    # Put rho (should be negative)
    put_rho = rho(100, 100, 0.05, 0.2, 1, option_type='put')
    assert isinstance(put_rho, float)
    assert put_rho < 0
    
    # Test with arrays
    s = np.array([90, 100, 110])
    k = np.array([100, 100, 100])
    call_rhos = rho(s, k, 0.05, 0.2, 1, option_type='call')
    assert isinstance(call_rhos, np.ndarray)
    assert call_rhos.shape == (3,)
    assert np.all(call_rhos >= 0)

def test_implied_volatility():
    """Test the implied_volatility function."""
    # Calculate a call price
    price = call_price(100, 100, 0.05, 0.2, 1)
    
    # Now find the implied volatility
    iv = implied_volatility(price, 100, 100, 0.05, 1, option_type='call')
    assert isinstance(iv, float)
    assert iv == pytest.approx(0.2, abs=0.01)
    
    # Calculate a put price
    price = put_price(100, 100, 0.05, 0.2, 1)
    
    # Now find the implied volatility
    iv = implied_volatility(price, 100, 100, 0.05, 1, option_type='put')
    assert isinstance(iv, float)
    assert iv == pytest.approx(0.2, abs=0.01)

def test_calculate_all_greeks():
    """Test the calculate_all_greeks function."""
    # Call greeks
    greeks = calculate_all_greeks(100, 100, 0.05, 0.2, 1, option_type='call')
    assert isinstance(greeks, dict)
    assert 0.5 <= greeks["delta"] <= 0.7
    assert greeks["gamma"] > 0
    assert greeks["theta"] < 0
    assert greeks["vega"] > 0
    assert greeks["rho"] > 0
    
    # Put greeks
    greeks = calculate_all_greeks(100, 100, 0.05, 0.2, 1, option_type='put')
    assert isinstance(greeks, dict)
    assert -0.5 >= greeks["delta"] >= -0.7
    assert greeks["gamma"] > 0
    assert greeks["theta"] < 0
    assert greeks["vega"] > 0
    assert greeks["rho"] < 0

def test_calculate_iv_surface():
    """Test the calculate_iv_surface function."""
    # Create some sample market prices
    strikes = np.array([90, 100, 110])
    call_prices = np.array([15, 10, 5])
    put_prices = np.array([5, 10, 15])
    
    # Calculate IV surface
    call_ivs, put_ivs = calculate_iv_surface(
        call_prices, put_prices, 100, strikes, 0.05, 1
    )
    
    assert isinstance(call_ivs, np.ndarray)
    assert isinstance(put_ivs, np.ndarray)
    assert call_ivs.shape == (3,)
    assert put_ivs.shape == (3,)
    assert np.all(call_ivs > 0)
    assert np.all(put_ivs > 0) 
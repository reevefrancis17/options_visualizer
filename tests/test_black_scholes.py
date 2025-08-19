"""
Unit tests for the Black-Scholes option pricing model.
"""
import pytest
import numpy as np
from backend.models.black_scholes import (
    call_price, put_price, delta, gamma, theta, vega, rho, 
    implied_volatility, calculate_all_greeks, calculate_iv_surface
)


def test_call_price():
    """Test the Black-Scholes call option price calculation."""
    # At-the-money call option with 1 year to expiry
    price = call_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    # Expected price based on Black-Scholes formula
    assert 10.0 < price < 11.0  # Approximate range for the given parameters


def test_put_price():
    """Test the Black-Scholes put option price calculation."""
    # At-the-money put option with 1 year to expiry
    price = put_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    # Expected price based on Black-Scholes formula
    assert 5.0 < price < 6.0  # Approximate range for the given parameters


def test_put_call_parity():
    """Test the put-call parity relationship."""
    S = 100  # Stock price
    K = 100  # Strike price
    T = 1    # Time to expiry (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    
    c = call_price(S, K, T, r, sigma)
    p = put_price(S, K, T, r, sigma)
    
    # Put-call parity: c - p = S - K*exp(-r*T)
    expected_difference = S - K * np.exp(-r * T)
    actual_difference = c - p
    
    assert abs(actual_difference - expected_difference) < 1e-10


def test_delta():
    """Test the delta calculation."""
    # Call delta should be between 0 and 1
    call_delta = delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert 0.5 < call_delta < 0.7  # ATM call delta is approximately 0.5-0.6
    
    # Put delta should be between -1 and 0
    put_delta = delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert -0.3 > put_delta > -0.5  # ATM put delta is approximately -0.36


def test_gamma():
    """Test the gamma calculation."""
    # Gamma should be positive for both calls and puts
    call_gamma = gamma(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    put_gamma = gamma(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    
    # Gamma should be the same for calls and puts with the same parameters
    assert abs(call_gamma - put_gamma) < 1e-10
    
    # Gamma should be positive
    assert call_gamma > 0


def test_theta():
    """Test the theta calculation."""
    # Theta should be negative for both calls and puts (time decay)
    call_theta = theta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    put_theta = theta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    
    # Theta should be negative (options lose value over time)
    assert call_theta < 0
    assert put_theta < 0


def test_vega():
    """Test the vega calculation."""
    # Vega should be positive for both calls and puts
    call_vega = vega(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    put_vega = vega(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    
    # Vega should be the same for calls and puts with the same parameters
    assert abs(call_vega - put_vega) < 1e-10
    
    # Vega should be positive
    assert call_vega > 0


def test_rho():
    """Test the rho calculation."""
    # Rho should be positive for calls and negative for puts
    call_rho = rho(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    put_rho = rho(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    
    # Call rho should be positive
    assert call_rho > 0
    
    # Put rho should be negative
    assert put_rho < 0


def test_implied_volatility():
    """Test the implied volatility calculation."""
    # Calculate a call price with known parameters
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    
    price = call_price(S, K, T, r, sigma)
    
    # Now calculate the implied volatility from this price
    calculated_iv = implied_volatility(price, S, K, T, r, option_type="call")
    
    # The calculated IV should be close to the original sigma
    assert abs(calculated_iv - sigma) < 1e-4


def test_calculate_all_greeks():
    """Test the function that calculates all Greeks at once."""
    greeks = calculate_all_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    
    # Check that all Greeks are calculated
    assert "delta" in greeks
    assert "gamma" in greeks
    assert "theta" in greeks
    assert "vega" in greeks
    assert "rho" in greeks
    
    # Check that the values are reasonable
    assert 0.5 < greeks["delta"] < 0.7  # ATM call delta
    assert greeks["gamma"] > 0  # Gamma is positive
    assert greeks["theta"] < 0  # Theta is negative
    assert greeks["vega"] > 0   # Vega is positive
    assert greeks["rho"] > 0    # Call rho is positive 

@pytest.mark.parametrize("S, K, T, r, sigma, expected", [
    (100, 100, 1, 0.05, 0, 0),  # sigma=0
    (100, 90, 0, 0.05, 0.2, 10),  # T=0
    (100, 0, 1, 0.05, 0.2, 100),  # K=0
    (0, 100, 1, 0.05, 0.2, 0),  # S=0
])
def test_call_price_edge_cases(S, K, T, r, sigma, expected):
    assert call_price(S, K, T, r, sigma) == expected

@pytest.mark.parametrize("S, K, T, r, sigma, expected", [
    (100, 100, 1, 0.05, 0, 0),  # sigma=0
    (90, 100, 0, 0.05, 0.2, 10),  # T=0
    (100, 0, 1, 0.05, 0.2, 0),  # K=0
    (0, 100, 1, 0.05, 0.2, np.exp(-0.05 * 1) * 100),  # S=0
])
def test_put_price_edge_cases(S, K, T, r, sigma, expected):
    assert put_price(S, K, T, r, sigma) == expected

@pytest.mark.parametrize("S, K, T, r, sigma, option_type, expected", [
    (100, 100, 0, 0.05, 0.2, 'call', 0.0),
    (110, 100, 0, 0.05, 0.2, 'call', 1.0),
    (90, 100, 0, 0.05, 0.2, 'call', 0.0),
    (100, 100, 0, 0.05, 0.2, 'put', 0.0),
    (90, 100, 0, 0.05, 0.2, 'put', -1.0),
    (110, 100, 0, 0.05, 0.2, 'put', 0.0),
])
def test_greeks_edge_cases(S, K, T, r, sigma, option_type, expected):
    assert abs(delta(S, K, T, r, sigma, option_type) - expected) < 0.1  # Loose assertion

# Similar for other greeks

def test_implied_volatility_edge_cases():
    iv = implied_volatility( -1, 100, 100, 1, 0.05, 'call')  # Negative price
    assert np.isnan(iv)

    iv = implied_volatility(1e-7, 100, 100, 1, 0.05, 'call')
    assert iv == 0.01

    # Test market_price &lt; intrinsic for ITM
    iv = implied_volatility(9.9, 110, 100, 1, 0.05, 'call')
    assert np.isnan(iv)

def test_calculate_iv_surface():
    market_prices = np.array([[5.0, 6.0], [3.0, 4.0]])
    strikes = [95, 105]
    expiries = [0.5, 1.0]
    S = 100
    r = 0.05
    option_type = 'call'
    iv_surface = calculate_iv_surface(market_prices, strikes, expiries, S, r, option_type)
    assert iv_surface.shape == market_prices.shape
    assert np.all(iv_surface > 0) 
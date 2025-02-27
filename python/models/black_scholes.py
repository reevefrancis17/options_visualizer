#black scholes model
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

def d1(S, K, T, r, sigma):
    """Compute d1 component of Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    """Compute d2 component of Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price."""
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

def put_price(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price."""
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

def delta(S, K, T, r, sigma, option_type):
    """Calculate option Delta."""
    d1_val = d1(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(d1_val)
    elif option_type == 'put':
        return norm.cdf(d1_val) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(S, K, T, r, sigma):
    """Calculate option Gamma."""
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type):
    """Calculate option Theta."""
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    term1 = - (S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2_val)
        return term1 - term2
    elif option_type == 'put':
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
        return term1 + term2
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def vega(S, K, T, r, sigma):
    """Calculate option Vega."""
    d1_val = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d1_val)

def rho(S, K, T, r, sigma, option_type):
    """Calculate option Rho."""
    d2_val = d2(S, K, T, r, sigma)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2_val)
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2_val)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def implied_volatility(market_price, S, K, T, r, option_type, initial_guess=0.2, max_iter=100, tol=1e-6):
    """Calculate Implied Volatility using Newton-Raphson method."""
    def price_difference(sigma):
        if option_type == 'call':
            return call_price(S, K, T, r, sigma) - market_price
        elif option_type == 'put':
            return put_price(S, K, T, r, sigma) - market_price
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    try:
        iv = newton(price_difference, initial_guess, maxiter=max_iter, tol=tol)
        return iv if iv > 0 else np.nan  # Return NaN if IV is negative
    except RuntimeError:
        return np.nan
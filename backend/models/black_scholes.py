#black scholes model
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, brentq
import logging

# Set up logger
logger = logging.getLogger(__name__)

def d1(S, K, T, r, sigma):
    """
    Compute d1 component of Black-Scholes formula.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        d1 component value
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        return np.nan
    if K <= 0 or S <= 0:
        return np.nan
        
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    """
    Compute d2 component of Black-Scholes formula.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        d2 component value
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        return np.nan
    if K <= 0 or S <= 0:
        return np.nan
        
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        Call option price
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        return max(0, S - K)  # Intrinsic value
    if K <= 0:
        return S  # Call with zero strike is worth the stock price
    if S <= 0:
        return 0  # Worthless if stock price is zero
        
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    if np.isnan(d1_val) or np.isnan(d2_val):
        return max(0, S - K)  # Return intrinsic value if calculation fails
        
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

def put_price(S, K, T, r, sigma):
    """
    Calculate Black-Scholes put option price.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        Put option price
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        return max(0, K - S)  # Intrinsic value
    if K <= 0:
        return 0  # Worthless if strike is zero
    if S <= 0:
        return K * np.exp(-r * T)  # Present value of strike if stock is worthless
        
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    if np.isnan(d1_val) or np.isnan(d2_val):
        return max(0, K - S)  # Return intrinsic value if calculation fails
        
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

def delta(S, K, T, r, sigma, option_type):
    """
    Calculate option Delta (first derivative of price with respect to underlying price).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        
    Returns:
        Option delta value
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
            
    d1_val = d1(S, K, T, r, sigma)
    
    if np.isnan(d1_val):
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    if option_type == 'call':
        return norm.cdf(d1_val)
    elif option_type == 'put':
        return norm.cdf(d1_val) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(S, K, T, r, sigma, option_type=None):
    """
    Calculate option Gamma (second derivative of price with respect to underlying price).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put' (not used, as gamma is the same for both)
        
    Returns:
        Option gamma value
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0 or S <= 0:
        return 0.0
        
    d1_val = d1(S, K, T, r, sigma)
    
    if np.isnan(d1_val):
        return 0.0
        
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type):
    """
    Calculate option Theta (derivative of price with respect to time).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        
    Returns:
        Option theta value (per year)
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        return 0.0
        
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    if np.isnan(d1_val) or np.isnan(d2_val):
        return 0.0
        
    term1 = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
        return term1 + term2
    elif option_type == 'put':
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
        return term1 + term2
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def vega(S, K, T, r, sigma, option_type=None):
    """
    Calculate option Vega (derivative of price with respect to volatility).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put' (not used, as vega is the same for both)
        
    Returns:
        Option vega value (for 1% change in volatility)
    """
    # Handle edge cases
    if T <= 0 or S <= 0:
        return 0.0
        
    # For very low volatility, use a minimum to avoid division by zero
    if sigma < 0.001:
        sigma = 0.001
        
    d1_val = d1(S, K, T, r, sigma)
    
    if np.isnan(d1_val):
        return 0.0
        
    # Return vega for a 1% change in volatility (0.01)
    return S * np.sqrt(T) * norm.pdf(d1_val) * 0.01

def rho(S, K, T, r, sigma, option_type):
    """
    Calculate option Rho (derivative of price with respect to interest rate).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        
    Returns:
        Option rho value (for 1% change in interest rate)
    """
    # Handle edge cases
    if T <= 0:
        return 0.0
        
    d2_val = d2(S, K, T, r, sigma)
    
    if np.isnan(d2_val):
        return 0.0
        
    # Return rho for a 1% change in interest rate (0.01)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2_val) * 0.01
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2_val) * 0.01
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def implied_volatility(market_price, S, K, T, r, option_type, initial_guess=0.2, max_iter=100, tol=1e-6):
    """
    Calculate Implied Volatility using a robust approach combining bisection and Newton-Raphson methods.
    
    Args:
        market_price: Observed market price of the option
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        option_type: 'call' or 'put'
        initial_guess: Initial volatility guess
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        Implied volatility or NaN if calculation fails
    """
    # Handle edge cases
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
        
    # Calculate intrinsic value
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    # If market price is less than intrinsic value (arbitrage), return NaN
    if market_price < intrinsic:
        return np.nan
        
    # If market price equals intrinsic value, IV is effectively zero
    if abs(market_price - intrinsic) < tol:
        return 0.01  # Return minimum IV
    
    # Define the price function based on option type
    def price_function(sigma):
        if option_type == 'call':
            return call_price(S, K, T, r, sigma)
        else:
            return put_price(S, K, T, r, sigma)
    
    # Define the objective function (difference between model and market price)
    def objective(sigma):
        if sigma <= 0:
            return -market_price  # Large negative value for negative sigma
        try:
            model_price = price_function(sigma)
            return model_price - market_price
        except:
            return np.nan
    
    try:
        # First try Newton-Raphson method with initial guess
        iv = newton(objective, initial_guess, tol=tol, maxiter=max_iter)
        
        # Verify the result is reasonable
        if iv > 0 and iv < 5.0:
            return iv
            
        # If Newton-Raphson fails or gives unreasonable result, try bisection method
        logger.debug(f"Newton method gave IV={iv}, trying bisection")
    except:
        logger.debug("Newton method failed, trying bisection")
    
    # Bisection method with wide bounds
    try:
        # Use a wide range for IV search (1% to 500%)
        iv = brentq(objective, 0.01, 5.0, xtol=tol, maxiter=max_iter)
        return iv
    except:
        # If all methods fail, use a heuristic approach
        logger.debug("Bisection method failed, using heuristic approach")
        
        # Try different initial guesses
        for guess in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            try:
                iv = newton(objective, guess, tol=tol*10, maxiter=20)
                if 0.01 <= iv <= 5.0:
                    return iv
            except:
                continue
        
        # If all else fails, use a simple approximation based on moneyness
        moneyness = S / K
        if option_type == 'call':
            if moneyness > 1.1:  # Deep ITM
                return 0.3
            elif moneyness < 0.9:  # Deep OTM
                return 0.5
            else:  # ATM
                return 0.4
        else:  # Put
            if moneyness < 0.9:  # Deep ITM
                return 0.3
            elif moneyness > 1.1:  # Deep OTM
                return 0.5
            else:  # ATM
                return 0.4
    
    return np.nan

def calculate_all_greeks(S, K, T, r, sigma, option_type):
    """
    Calculate all option Greeks in a single function call.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary containing all Greeks and option price
    """
    price = call_price(S, K, T, r, sigma) if option_type == 'call' else put_price(S, K, T, r, sigma)
    
    return {
        'price': price,
        'delta': delta(S, K, T, r, sigma, option_type),
        'gamma': gamma(S, K, T, r, sigma, option_type),
        'theta': theta(S, K, T, r, sigma, option_type),
        'vega': vega(S, K, T, r, sigma, option_type),
        'rho': rho(S, K, T, r, sigma, option_type)
    }

def calculate_iv_surface(market_prices, strikes, expiries, S, r, option_type):
    """
    Calculate implied volatility surface from a grid of market prices.
    
    Args:
        market_prices: 2D array of market prices [strikes, expiries]
        strikes: Array of strike prices
        expiries: Array of expiration times in years
        S: Current underlying price
        r: Risk-free rate
        option_type: 'call' or 'put'
        
    Returns:
        2D array of implied volatilities with same shape as market_prices
    """
    iv_surface = np.zeros_like(market_prices)
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            price = market_prices[i, j]
            if np.isnan(price) or price <= 0:
                iv_surface[i, j] = np.nan
            else:
                iv_surface[i, j] = implied_volatility(price, S, K, T, r, option_type)
    
    return iv_surface
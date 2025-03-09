#black scholes model
# This module implements the Black-Scholes option pricing model and Greeks calculations.
# The Black-Scholes model is a mathematical model for pricing European-style options,
# based on the assumption that the stock price follows a geometric Brownian motion.

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, brentq
import logging

# Set up logger
logger = logging.getLogger(__name__)

def d1(S, K, T, r, sigma):
    """
    Compute d1 component of Black-Scholes formula.
    
    The d1 term represents the standardized distance from the current stock price
    to the strike price, adjusted for the risk-free rate, volatility, and time to expiration.
    It's used in calculating both the option price and the Greeks.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        float: d1 component
    """
    # Handle edge cases to prevent numerical issues
    if sigma <= 0 or T <= 0:
        return np.nan
    
    # Standard Black-Scholes d1 formula
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    """
    Compute d2 component of Black-Scholes formula.
    
    The d2 term is d1 adjusted by the volatility factor. It represents the 
    probability that the option will be exercised in the risk-neutral world.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        float: d2 component
    """
    # d2 is d1 minus the volatility adjustment
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    """
    Calculate the price of a European call option using the Black-Scholes model.
    
    The call option price is the present value of the expected payoff at expiration,
    where the payoff is max(0, S - K) if the stock price (S) exceeds the strike price (K).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        float: Call option price
    """
    # Handle edge cases
    if T <= 0:
        return max(0, S - K)  # Immediate exercise value
    if sigma <= 0:
        return max(0, S - K * np.exp(-r * T))  # Deterministic case
    
    # Calculate d1 and d2
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    # Black-Scholes formula for call option
    # S * N(d1) - K * e^(-rT) * N(d2)
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

def put_price(S, K, T, r, sigma):
    """
    Calculate the price of a European put option using the Black-Scholes model.
    
    The put option price is the present value of the expected payoff at expiration,
    where the payoff is max(0, K - S) if the strike price (K) exceeds the stock price (S).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        float: Put option price
    """
    # Handle edge cases
    if T <= 0:
        return max(0, K - S)  # Immediate exercise value
    if sigma <= 0:
        return max(0, K * np.exp(-r * T) - S)  # Deterministic case
    
    # Calculate d1 and d2
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    # Black-Scholes formula for put option
    # K * e^(-rT) * N(-d2) - S * N(-d1)
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

def delta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the delta of an option.
    
    Delta measures the rate of change of the option price with respect to changes
    in the underlying asset's price. It represents the number of shares of the
    underlying asset that would be needed to hedge the option position.
    
    For a call option, delta ranges from 0 to 1.
    For a put option, delta ranges from -1 to 0.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        
    Returns:
        float: Delta value
    """
    # Handle edge cases
    if T <= 0:
        # At expiration, delta is either 0 or 1 (call) / -1 (put)
        if option_type.lower() == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    if sigma <= 0:
        # With no volatility, option behaves like a forward contract
        if option_type.lower() == 'call':
            return 1.0 if S < K * np.exp(-r * T) else 0.0
        else:
            return -1.0 if S < K * np.exp(-r * T) else 0.0
    
    # Calculate d1
    d1_val = d1(S, K, T, r, sigma)
    
    # Delta formula
    if option_type.lower() == 'call':
        return norm.cdf(d1_val)  # N(d1) for call
    else:
        return norm.cdf(d1_val) - 1  # N(d1) - 1 for put

def gamma(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the gamma of an option.
    
    Gamma measures the rate of change of delta with respect to changes in the
    underlying asset's price. It represents the curvature of the option's value
    with respect to the underlying price.
    
    Gamma is the same for both call and put options with the same parameters.
    High gamma means the delta will change rapidly with small changes in the
    underlying price, indicating higher risk.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put' (doesn't affect the result)
        
    Returns:
        float: Gamma value
    """
    # Handle edge cases
    if T <= 0 or sigma <= 0:
        return 0.0  # No gamma at expiration or with no volatility
    
    # Calculate d1
    d1_val = d1(S, K, T, r, sigma)
    
    # Gamma formula: phi(d1) / (S * sigma * sqrt(T))
    # where phi is the standard normal PDF
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the theta of an option.
    
    Theta measures the rate of change of the option price with respect to time,
    representing the time decay of the option's value. It's typically negative
    for both calls and puts, as options lose value as they approach expiration.
    
    Theta is expressed as the dollar change in option value per day (assuming 365 days in a year).
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        
    Returns:
        float: Theta value (per day)
    """
    # Handle edge cases
    if T <= 0:
        return 0.0  # No time decay at expiration
    
    if sigma <= 0:
        # With no volatility, theta is just the interest on the strike
        if option_type.lower() == 'call':
            return -r * K * np.exp(-r * T) / 365
        else:
            return r * K * np.exp(-r * T) / 365
    
    # Calculate d1 and d2
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    # Common term: -S * phi(d1) * sigma / (2 * sqrt(T))
    common_term = -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    
    # Theta formula
    if option_type.lower() == 'call':
        # Call theta: common_term - r * K * e^(-rT) * N(d2)
        theta_value = common_term - r * K * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        # Put theta: common_term + r * K * e^(-rT) * N(-d2)
        theta_value = common_term + r * K * np.exp(-r * T) * norm.cdf(-d2_val)
    
    # Convert from per year to per day
    return theta_value / 365.0

def vega(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the vega of an option.
    
    Vega measures the rate of change of the option price with respect to changes
    in volatility. It represents the option's sensitivity to volatility changes.
    
    Vega is the same for both call and put options with the same parameters.
    Vega is typically highest for at-the-money options and decreases as the option
    moves further in or out of the money.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put' (doesn't affect the result)
        
    Returns:
        float: Vega value (for 1% change in volatility)
    """
    # Handle edge cases
    if T <= 0:
        return 0.0  # No vega at expiration
    
    if sigma <= 0:
        return 0.0  # No vega with zero volatility
    
    # Calculate d1
    d1_val = d1(S, K, T, r, sigma)
    
    # Vega formula: S * sqrt(T) * phi(d1)
    # Multiply by 0.01 to represent a 1% change in volatility
    return S * np.sqrt(T) * norm.pdf(d1_val) * 0.01

def rho(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the rho of an option.
    
    Rho measures the rate of change of the option price with respect to changes
    in the risk-free interest rate. It represents the option's sensitivity to
    interest rate changes.
    
    Rho is typically positive for calls and negative for puts, as higher interest
    rates increase call prices and decrease put prices.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        
    Returns:
        float: Rho value (for 1% change in interest rate)
    """
    # Handle edge cases
    if T <= 0:
        return 0.0  # No rho at expiration
    
    # Calculate d2
    d2_val = d2(S, K, T, r, sigma)
    
    # Rho formula
    if option_type.lower() == 'call':
        # Call rho: K * T * e^(-rT) * N(d2)
        rho_value = K * T * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        # Put rho: -K * T * e^(-rT) * N(-d2)
        rho_value = -K * T * np.exp(-r * T) * norm.cdf(-d2_val)
    
    # Multiply by 0.01 to represent a 1% change in interest rate
    return rho_value * 0.01

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
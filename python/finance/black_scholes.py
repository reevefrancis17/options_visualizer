import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize Black-Scholes calculator
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self._calculate_d1_d2()

    def _calculate_d1_d2(self):
        """Calculate d1 and d2 parameters"""
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) 
                  * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def call_price(self):
        """Calculate call option price"""
        return (self.S * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))

    def put_price(self):
        """Calculate put option price"""
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                self.S * norm.cdf(-self.d1)) 
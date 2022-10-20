import numpy as np
from scipy import stats as sps
from scipy.optimize import root_scalar

from typing import Union, Tuple, Optional


def black_scholes(K:Union[float, np.ndarray], F:Union[float, np.ndarray], T:Union[float, np.ndarray], r:float, vol:float) -> Union[float, np.ndarray]:
    """
        This function returns the price of a call option with passed parameters:

        Args:
            K(Union[float, np.ndarray]): strike or array of strikes, 
            F(Union[float, np.ndarray]): underlying price or array of underlying prices, 
            T(Union[float, np.ndarray]): expiration time or array of expiration times, 
            r(float): interest rate, 
            vol(float): Volatility in BS model

        Returns:
            call_price(Union[float, np.ndarray]): call price
    """
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) \
                / (vol * np.sqrt(T) + 1e-10)
    d2 = d1 - vol * np.sqrt(T)
    D = np.exp(-r * T)
    call_price =  D *  ( F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2) )
    return call_price


def black_scholes_vega(K:Union[float, np.ndarray], F:Union[float, np.ndarray], T:Union[float, np.ndarray], r:float, vol:float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
        This function returns the price of a call option and its vega:

        Args:
            K(Union[float, np.ndarray]): strike or array of strikes, 
            F(Union[float, np.ndarray]): underlying price or array of underlying prices, 
            T(Union[float, np.ndarray]): expiration time or array of expiration times, 
            r(float): interest rate, 
            vol(float): Volatility in BS model

        Returns:
            call_price(Union[float, np.ndarray]): call price
            vega(Union[float, np.ndarray]): vega of a call option
    """
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) \
                / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    D = np.exp(-r * T)
    call_price =  D *  ( F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2) )
    
    vega = D * F * sps.norm.pdf(d1) * np.sqrt(T)
    return call_price, vega


def implied_volatility_helper(C:float, K:float, F:float, T:float, r:float) -> Optional[float]:
    """
        This function returns implied volatility of a call option with price C and parameters K, F, T, r

        Args:
            C(float): option price
            K(float): strike
            F(float): futures price
            T(float): expiration time
            r(float): interest rate
        
        Returns:
            vol(Union[float, np.nan]): implied volatility. If there is no root, this function returns np.nan
    """
    def foo(vol):
        C_tmp = black_scholes(K, F, T, r, vol)
        return C_tmp - C
    v0 = 1e-15
    v1 = 100
    vol = np.nan
    if foo(v0) <= 0 and foo(v1) >= 0:
        vol = root_scalar(foo, bracket=[v0, v1], method='bisect').root
    return vol
    

def implied_volatility(C:Union[float, np.ndarray], K:Union[float, np.ndarray], 
                      F:Union[float, np.ndarray], T:Union[float, np.ndarray], r:float) -> Union[float, np.ndarray]:
    """
        This function returns implied volatility of call options with prices C and parameters K, F, T, r

        Args:
            C(Union[float, np.ndarray]): option price
            K(Union[float, np.ndarray]): strike
            F(Union[float, np.ndarray]): futures price
            T(Union[float, np.ndarray]): expiration time
            r(float): interest rate
        
        Returns:
            vol(Union[float, np.ndarray]): implied volatility
    """

    if isinstance(C, float):
        return implied_volatility_helper(C, K, F, T, r)
    if isinstance(C, np.ndarray):

        assert isinstance(K, np.ndarray)
        assert len(K) == len(C)
        
        if isinstance(F, np.ndarray):
            assert len(F) == len(C)
        else:
            #repeat F len(C) times
            F = np.ones_like(C) * F
        if isinstance(T, np.ndarray):
            assert len(T) == len(C)
        else:
            #repeat T len(C) times
            T = np.ones_like(C) * T
        
    result = [ implied_volatility_helper(c, k, f, t, r) for c, k, f, t in zip(C, K, F, T) ]
    return np.asarray(result)
    
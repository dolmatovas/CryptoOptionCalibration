from typing import Optional, Union, Tuple, List

import numpy as np
from scipy import stats as sps
from scipy.optimize import root_scalar

from black_scholes import black_scholes, black_scholes_vega


def sabr_approx(K:Union[float, np.ndarray], F0:Union[float, np.ndarray], T:Union[float, np.ndarray], 
                r:float, sig0:float, alpha:float, beta:float, rho:float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
        This function returns approximation for option price and impied volatility in SABR model
        Args:
            K(Union[float, np.ndarray]): strikes, 
            F0(Union[float, np.ndarray]): underlying futures price, 
            T(Union[float, np.ndarray]): expiration time, 
            r(float): interest rate, 
            sig0(float): initial volatility,
            alpha(float): alpha parameter from SABR model, vol of vol, 
            beta(float): beta parameter from SABR model, volatility elasticity, 
            rho(float): correlation between Wienner processes
        Returns:
            C(Union[float, np.ndarray]): call price
            vol(Union[float, np.ndarray]): implied volatility
    """
    #mid price
    Fm = np.sqrt( F0 * K )
    #small parameter
    eps = alpha ** 2 * T
    
    zeta = alpha / (sig0) / (1 - beta) * ( F0 ** (1-beta) - K ** (1-beta) )
    D = np.log( (np.sqrt(1 - 2 * rho * zeta + zeta ** 2) + zeta - rho) / (1-rho) )
    
    q1 = Fm ** (2 * beta - 2) * ( sig0 * (1-beta) / alpha ) ** 2 / 24
    q2 = Fm ** (beta - 1) * rho * sig0 * beta / (4 * alpha) 
    q3 = (2 - 3 * rho ** 2) / 24
    
    vol = alpha * np.log(F0 / K) / D * ( 1 + (q1 + q2 + q3) * eps )
    C = black_scholes(K, F0, T, r, vol)
    return C, vol


def sabr_approx_derivatives(K:Union[float, np.ndarray], F0:Union[float, np.ndarray], T:Union[float, np.ndarray], 
                r:float, sig0:float, alpha:float, beta:float, rho:float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
        This function returns approximation for option price and impied volatility in SABR model as well as the derivatives with respect to this parameters
        Args:
            K(Union[float, np.ndarray]): strikes, 
            F0(Union[float, np.ndarray]): underlying futures price, 
            T(Union[float, np.ndarray]): expiration time, 
            r(float): interest rate, 
            sig0(float): initial volatility,
            alpha(float): alpha parameter from SABR model, vol of vol, 
            beta(float): beta parameter from SABR model, volatility elasticity, 
            rho(float): correlation between Wienner processes
        Returns:
            C(Union[float, np.ndarray]): call price
            vega(Union[float, np.ndarray]): vega
            vol(Union[float, np.ndarray]): implied volatility
            vol_sig0(Union[float, np.ndarray]): derivative with respect to sig0
            vol_alpha(Union[float, np.ndarray]): derivative with respect to alpha
            vol_beta(Union[float, np.ndarray]): derivative with respect to beta
            vol_rho(Union[float, np.ndarray]): derivative with respect to rho
    """
    
    #mid point
    Fm = np.sqrt( F0 * K )
    #small parameter
    eps = alpha ** 2 * T
    
    #zeta
    zeta = alpha / (sig0) / (1 - beta) * ( F0 ** (1-beta) - K ** (1-beta) )
    
    zeta_sig0 = -zeta / sig0
    
    zeta_alpha = zeta / alpha
    
    zeta_beta = zeta / (1 - beta) - alpha / (sig0 * (1-beta)) * \
    ( np.log(F0) * F0 ** (1-beta) - np.log(K) * K ** (1-beta))
    
    #D(zeta)
    sqr = np.sqrt(1 - 2 * rho * zeta + zeta ** 2)       
    
    D = np.log( (sqr + zeta - rho) / (1-rho) )
    
    D_zeta = 1 / sqr
    
    D_rho = 1 / (1-rho) - (sqr + zeta) / sqr / (sqr + zeta - rho)
    
    #
    q1 = Fm ** (2 * beta - 2) * ( sig0 * (1-beta) / alpha ) ** 2 / 24
    q2 = rho * sig0 * beta / 4 / alpha * Fm ** (beta - 1)
    q3 = (2 - 3 * rho ** 2) / 24
    
    #S
    S = 1 + eps * (q1 + q2 + q3)
    
    S_sig0 = eps / sig0 * (2 * q1 + q2)
    
    S_alpha = alpha * T * (q2 + 2 * q3)
    
    S_beta  = eps * ( 2 * q1 * ( 1 / (beta - 1)  + np.log(Fm)) + q2 * ( 1 / beta + np.log(Fm)) )
    
    S_rho = eps * (Fm ** (beta - 1) * sig0 * beta / (4 * alpha) - rho / 4 )
    
    #result
    vol = alpha * np.log(F0 / K) / D * S
    
    vol_sig0 = -vol / D *  D_zeta * zeta_sig0 + alpha * np.log(F0/K) / D  * S_sig0 
    
    vol_alpha = vol / alpha - vol / D * D_zeta * zeta_alpha + alpha * np.log(F0 / K) / D * S_alpha
    
    vol_beta = -vol / D * D_zeta * zeta_beta + alpha * np.log(F0/K) / D * S_beta
    
    vol_rho = -vol / D * D_rho + alpha * np.log(F0/K) / D * S_rho
    
    C, vega = black_scholes_vega(K, F0, T, r, vol)
    return C, vega, vol, vol_sig0, vol_alpha, vol_beta, vol_rho 
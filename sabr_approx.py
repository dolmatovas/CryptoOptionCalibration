from typing import Optional, Union, Tuple, List

import numpy as np
from scipy import stats as sps
from scipy.optimize import root_scalar

from black_scholes import black_scholes, black_scholes_vega


def sabr_approx(K:Union[float, np.ndarray], F:Union[float, np.ndarray], T:Union[float, np.ndarray], 
                r:float, alpha:float, v:float, beta:float, rho:float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
        This function returns approximation for option price and impied volatility in SABR model
        Args:
            K(Union[float, np.ndarray]): strikes, 
            F0(Union[float, np.ndarray]): underlying futures price, 
            T(Union[float, np.ndarray]): expiration time, 
            r(float): interest rate, 
            alpha(float): initial volatility,
            v(float): v parameter from SABR model, vol of vol, 
            beta(float): beta parameter from SABR model, volatility elasticity, 
            rho(float): correlation between Wienner processes
        Returns:
            C(Union[float, np.ndarray]): call price
            sig(Union[float, np.ndarray]): implied volatility
    """
    
    #mid price
    Fm = np.sqrt(F * K)
    
    q1 = (beta - 1) ** 2 * alpha ** 2 * Fm ** (2 * beta - 2) / 24
    q2 = rho * beta * alpha * v * Fm ** (beta - 1) / 4
    q3 = (2 - 3 * rho ** 2) / 24 * v ** 2
    
    S = 1 + T * (q1 + q2 + q3)
    
    zeta = v / alpha * Fm ** (1 - beta) * np.log(F / K)
    sqrt = np.sqrt(1 - 2 * rho * zeta + zeta ** 2) 
    X = np.log( (sqrt + zeta - rho) / (1-rho))
    
    D = Fm ** (1-beta) * ( 1 + (beta-1)**2/24 * (np.log(F/K))**2 + (beta-1)**4/1920 * (np.log(F/K))**4 )
    
    sig = alpha * S * zeta / D / X
    
    C = black_scholes(K, F, T, r, sig)
    return C, sig


def sabr_approx_derivatives(K:Union[float, np.ndarray], F:Union[float, np.ndarray], T:Union[float, np.ndarray], 
                r:float, alpha:float, v:float, beta:float, rho:float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
        This function returns approximation for option price and impied volatility in SABR model as well as the derivatives with respect to this parameters
        Args:
            K(Union[float, np.ndarray]): strikes, 
            F0(Union[float, np.ndarray]): underlying futures price, 
            T(Union[float, np.ndarray]): expiration time, 
            r(float): interest rate, 
            alpha(float): initial volatility,
            v(float): v parameter from SABR model, vol of vol, 
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
    Fm = np.sqrt(F * K)
    
    q1 = (beta - 1) ** 2 * alpha ** 2 * Fm ** (2 * beta - 2) / 24
    q2 = rho * beta * alpha * v * Fm ** (beta - 1) / 4
    q3 = (2 - 3 * rho ** 2) / 24 * v ** 2
    
    S = 1 + T * (q1 + q2 + q3)
    
    zeta = v / alpha * Fm ** (1 - beta) * np.log(F / K)
    sqrt = np.sqrt(1 - 2 * rho * zeta + zeta ** 2) 
    X = np.log( (sqrt + zeta - rho) / (1-rho))
    
    D = Fm ** (1-beta) * ( 1 + (beta-1)**2/24 * (np.log(F/K))**2 + (beta-1)**4/1920 * (np.log(F/K))**4 )
    
    sig = alpha * S * zeta / D / X
    
    X_zeta = 1 / sqrt
    
    S_alpha = T * (2 * q1 + q2) / alpha
    zeta_alpha = -zeta / alpha
    X_alpha = X_zeta * zeta_alpha
    
    S_rho = T * v * (beta  * alpha * Fm**(beta - 1) - rho * v) / 4
    X_rho = 1 / (1 - rho) - 1 / sqrt * (sqrt + zeta) / (sqrt + zeta - rho)
    
    S_v = T / v * (q2 + 2 * q3)
    zeta_v = zeta / v
    X_v = X_zeta * zeta_v
    
    zeta_beta = -np.log(Fm) * zeta
    X_beta = X_zeta * zeta_beta
    S_beta = T * (2  * q1 * (1/(beta-1)+np.log(Fm)) + q2 * (1/beta + np.log(Fm)) )
    D_beta = -np.log(Fm) * D + Fm**(1-beta) * ( (beta-1)/12 * (np.log(F/K))**2 + (beta-1)**3/480 * (np.log(F/K))**4 )
    
    logs_alpha = 1 / alpha + S_alpha / S + zeta_alpha / zeta - X_alpha / X
    logs_v     = S_v / S + zeta_v / zeta - X_v / X
    logs_beta  = S_beta / S - D_beta / D + zeta_beta / zeta - X_beta / X
    logs_rho   = S_rho / S - X_rho / X
    
    sig_alpha = sig * logs_alpha
    sig_v     = sig * logs_v
    sig_beta  = sig * logs_beta
    sig_rho   = sig * logs_rho
    
    C, vega = black_scholes_vega(K, F, T, r, sig)
    return C, vega, sig, sig_alpha, sig_v, sig_beta, sig_rho 
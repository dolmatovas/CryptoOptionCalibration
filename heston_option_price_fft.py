from typing import Optional, Union, Tuple, List

import numpy as np
import scipy.stats as sps

from heston_phi import heston_phi, heston_phi_derivatives
from heston_option_price import check_input_types, get_mesh

def heston_option_price_fft(S:Union[np.ndarray, float], K:Union[np.ndarray, float], T:Union[np.ndarray, float], 
                 Nu:int, r:float, heston_params:np.ndarray, is_call=True) -> np.ndarray:
    """
        This function calculatas option price in Heston model
        Args:
            S(Union[np.ndarray, float]): current stock price            
            K(Union[np.ndarray, float]): strike prices
            T(Union[np.ndarray, float]): expiration time
            Nu(int): number of intgration points
            r(float): interest rate
            heston_params(np.ndarray): parameters of Heston model, 
            is_call(bool): boolean flag, if true, returns prices of a call option, otherwise of a put option.

        Returns:
            option price.
    """

    #transform input to np.ndarray
    S, K, T = check_input_types(S, K, T)
    Nk = len(K)

    un, hn = get_mesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * T) / K).reshape(-1, 1)
    
    psi = np.ones((len(T), Nu), np.complex128)
    
    alpha = 1.0

    #размерность
    Ndim = len(heston_params) // 5
    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}
        _psi = heston_phi(un - (alpha + 1) * 1j, T.reshape(-1, 1), v, theta, rho, k, sig)
        psi *= _psi

    den = alpha ** 2 + alpha - un ** 2 + 1j * un * (2 * alpha + 1)
    F = (np.exp(1j * un * xn) * psi / den).real * hn
    
    integral = np.sum(F, axis=-1) / np.pi

    C = K * np.exp( -r * T + xn.reshape(-1) * (alpha + 1)) * integral.reshape(-1)
    
    if is_call:
        res = C
    else:
        res = C - S + K * np.exp(-r * T)
    return res


def heston_option_price_fft_derivatives(S:Union[np.ndarray, float], K:Union[np.ndarray, float], T:Union[np.ndarray, float], 
                 Nu:int, r:float, heston_params:np.ndarray, is_call=True) -> np.ndarray:
    """
        This function calculatas option price in Heston model
        Args:
            S(Union[np.ndarray, float]): current stock price            
            K(Union[np.ndarray, float]): strike prices
            T(Union[np.ndarray, float]): expiration time
            Nu(int): number of intgration points
            r(float): interest rate
            heston_params(np.ndarray): parameters of Heston model, 
            is_call(bool): boolean flag, if true, returns prices of a call option, otherwise of a put option.

        Returns:
            option price.
    """

    #transform input to np.ndarray
    S, K, T = check_input_types(S, K, T)
    N = len(K)

    un, hn = get_mesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * T) / K).reshape(-1, 1)
    
    psi = np.ones((len(T), Nu), np.complex128)
    
    alpha = 1.0

    #размерность
    Ndim = len(heston_params) // 5
    der  = np.zeros( (Ndim * 5, N, Nu), np.complex128)

    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]

        _psi, _der = heston_phi_derivatives(un - (alpha + 1) * 1j, T.reshape(-1, 1), v, theta, rho, k, sig )

        psi *= _psi
        der[5 * i : 5 * i + 5, :] = _der

    den = alpha ** 2 + alpha - un ** 2 + 1j * un * (2 * alpha + 1)
    
    
    F = np.exp(1j * un * xn) * psi / (den) * hn
    integral = np.sum(F.real, axis=-1) / np.pi
    integral_derivatives = np.sum( (F * der).real, axis=-1 ) / np.pi

    C = K * np.exp( -r * T + xn.reshape(-1) * (alpha + 1)) * integral
    derivatives = K * np.exp( -r * T + xn.reshape(-1) * (alpha + 1)) * integral_derivatives
    
    if is_call:
        res = C
    else:
        res = C - S + K * np.exp(-r * T)
    return res, derivatives
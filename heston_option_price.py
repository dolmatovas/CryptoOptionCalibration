from typing import Optional, Union, Tuple, List

import numpy as np
import scipy.stats as sps

from heston_phi import heston_phi, heston_phi_derivatives

from numbers import Number


def check_input_types(S:Union[np.ndarray, float], K:Union[np.ndarray, float], T:Union[np.ndarray, float]):
    """
        this function just checks if hte input parameters and np.ndarray, and if it float, wrapps them to array
    """
    if isinstance(K, np.ndarray):
        if isinstance(S, np.ndarray):
            assert len(S) == len(K)
        else:
            assert isinstance(S, Number)
            S = np.ones_like(K) * S
        if isinstance(T, np.ndarray):
            assert len(T) == len(K)
        else:
            assert isinstance(T, Number)
            T = np.ones_like(K) * T     
    
    elif isinstance(S, np.ndarray):
        if isinstance(K, np.ndarray):
            assert len(K) == len(S)
        else:
            assert isinstance(K, Number)
            K = np.ones_like(S) * K        
        if isinstance(T, np.ndarray):
            assert len(T) == len(S)
        else:
            assert isinstance(T, Number)
            T = np.ones_like(S) * T
    
    elif isinstance(T, np.ndarray):
        if isinstance(S, np.ndarray):
            assert len(S) == len(T)
        else:
            assert isinstance(S, Number)
            S = np.ones_like(T) * S
        if isinstance(K, np.ndarray):
            assert len(K) == len(T)
        else:
            assert isinstance(K, Number)
            K = np.ones_like(T) * K           
    
    else:
        assert isinstance(K, Number) and isinstance(S, Number) and isinstance(T, Number)
        K = np.asarray([K])
        S = np.asarray([S])
        T = np.asarray([T])


    return S, K, T



def get_mesh(Nu:int) -> Tuple[np.ndarray, np.ndarray]:
    tn = np.linspace(0, 1, (Nu // 2) + 1)
    h = tn[1] - tn[0]
    tn = tn[:-1] + h / 2.0
    
    a = 30
    n = 1
    
    u1 = a * (tn ** n)
    h1 = h * a * n * (tn ** (n-1))
    
    u2 = a + a * n * (-np.log(1 - tn))
    h2 = h * a * n / (1 - tn)
    
    un = np.concatenate((u1, u2))
    hn = np.concatenate((h1, h2))
    return un, hn


def heston_option_price(S:Union[np.ndarray, float], K:Union[np.ndarray, float], T:Union[np.ndarray, float], 
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

    un, hn = get_mesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * T) / K).reshape(-1, 1)
    
    phi1 = np.ones((len(T), Nu), np.complex128)
    phi2 = np.ones((len(T), Nu), np.complex128)
    
    #размерность
    Ndim = len(heston_params) // 5
    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}

        _phi1 = heston_phi(un, T.reshape(-1, 1), v, theta, rho, k, sig)
        _phi2 = heston_phi(un - 1j, T.reshape(-1, 1), v, theta, rho, k, sig)
        
        phi1 *= _phi1
        phi2 *= _phi2
        
        
    F1 = np.exp(1j * un * xn) * phi1 / (1j * un)
    F2 = np.exp(1j * un * xn) * phi2 / (1j * un)
    
    F1 = F1.real * hn
    F2 = F2.real * hn
    
    integral1 = np.sum(F1, axis=-1) / np.pi
    integral2 = np.sum(F2, axis=-1) / np.pi
    if is_call:
        P1 = 0.5 + integral1
        P2 = 0.5 + integral2
        res = S * P2 - np.exp(-r * T) * K * P1
    else:
        P1 = 0.5 - integral1
        P2 = 0.5 - integral2
        res = np.exp(-r * T) * K * P1 - S * P2
    return res



def heston_option_price_derivatives(S:Union[np.ndarray, float], K:Union[np.ndarray, float], T:Union[np.ndarray, float], 
                            Nu:int, r:float, heston_params:np.ndarray, is_call=True) -> Tuple[np.ndarray, np.ndarray]:
    """
        This function calculatas option price and it derivatives with respect to parameters of the heston model

        Args:
            K(np.ndarray): strike prices
            S(np.ndarray): current stock price            
            T(np.ndarray): expiration time
            Nu(int): number of intgration points
            r(float): interest rate
            heston_params(np.ndarray): parameters of Heston model, 
            is_call(bool): boolean flag, if true, returns prices of a call option, otherwise of a put option.

        Returns:
            res(np.ndarray): option price
            res_der(np.ndarray): derivatives
            
            res_der.shape = len(heston_params), len(res)
    """

    #transform input to np.ndarray
    S, K, T = check_input_types(S, K, T)

    N = len(K)
    Ndim = len(heston_params) // 5

    un, hn = get_mesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * T) / K).reshape(-1, 1)
    
    phi1 = np.ones((N, Nu), np.complex128)
    phi2 = np.ones((N, Nu), np.complex128)

    der1 = np.zeros( (Ndim * 5, N, Nu), np.complex128)
    der2 = np.zeros( (Ndim * 5, N, Nu), np.complex128)

    
    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]

        _phi1, _der1 = heston_phi_derivatives(un     , T.reshape(-1, 1), v, theta, rho, k, sig )
        _phi2, _der2 = heston_phi_derivatives(un - 1j, T.reshape(-1, 1), v, theta, rho, k, sig )
        
        assert _phi1.shape == (N, Nu)
        assert _der1.shape == (5, N, Nu)

        phi1 = phi1 * _phi1
        phi2 = phi2 * _phi2
        
        der1[5 * i : 5 * i + 5, :] = _der1
        der2[5 * i : 5 * i + 5, :] = _der2

    F1 = np.exp(1j * un * xn) * phi1 / (1j * un) * hn
    F2 = np.exp(1j * un * xn) * phi2 / (1j * un) * hn

    integral1 = np.sum(F1.real, axis=-1) / np.pi
    integral2= np.sum(F2.real, axis=-1) / np.pi

    integral_derivatives1 = np.sum( (F1 * der1).real, axis=-1 ) / np.pi
    integral_derivatives2 = np.sum( (F2 * der2).real, axis=-1 ) / np.pi
    assert integral_derivatives1.shape == (5 * Ndim, N)

    if is_call:
        P1 = 0.5 + integral1
        P2 = 0.5 + integral2
        res = S * P2 - np.exp(-r * T) * K * P1
    else:
        P1 = 0.5 - integral1
        P2 = 0.5 - integral2
        res = np.exp(-r * T) * K * P1 - S * P2
    
    res_der = S * integral_derivatives2 - np.exp(-r * T) * K * integral_derivatives1 
    assert res_der.shape == (5 * Ndim, N)
    return res, res_der
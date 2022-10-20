from typing import Optional, Union, Tuple, List

import numpy as np
import scipy.stats as sps

from numba import njit

from heston_phi import phi, phi_derivatives


def getMesh(Nu):
    tn = np.linspace(0, 1, (Nu // 2) + 1)
    h = tn[1] - tn[0]
    tn = tn[:-1] + h / 2.0
    
    a = 30
    n = 1
    f = lambda t: a * (t ** n)
    df = lambda t: a * n * (t ** (n-1))
    
    g = lambda t: -np.log(1 - t)
    dg = lambda t: 1 / (1 - t)
    
    u1 = f(tn)
    h1 = h * df(tn)
    
    u2 = a + df(1.0) * g(tn)
    h2 = h * df(1.0) * dg(tn)
    
    un = np.r_[u1, u2]
    hn = np.r_[h1, h2]
    return un, hn


def heston_option_price(S:Union[np.ndarray, float], K:Union[np.ndarray, float], tau:Union[np.ndarray, float], 
                 Nu:int, r:float, heston_params:np.ndarray, isCall=True) -> np.ndarray:
    '''
        return option price in Double heston model

        S -- current stock price. if S is np.ndarray, S should have the same length as K

        K -- strikes.

        tau -- time to expiration. If tau is np.ndarray, tau should have the same length as K

        Nu -- number of points in fourier integral

        r -- interest rate

        heston_params -- array of heston params. 

        isCall -- boolean flag.
        
        output shape
        res.shape == (len(K), )
    '''   

    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    if isinstance(tau, np.ndarray):
        assert len(tau) == len(K)
    else:
        tau = np.asarray([tau])
    if isinstance(S, np.ndarray):
        assert len(S) == len(K)
    else:
        S = np.asarray([S])

    Nk = len(K)

    un, hn = getMesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * tau) / K).reshape(-1, 1)
    
    phi1 = np.ones((len(tau), Nu), complex)
    phi2 = np.ones((len(tau), Nu), complex)
    
    #размерность
    Ndim = len(heston_params) // 5
    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}

        _phi1 = phi(un, tau.reshape(-1, 1), **params)
        _phi2 = phi(un - 1j, tau.reshape(-1, 1), **params)
        
        phi1 *= _phi1
        phi2 *= _phi2
        
        
    F1 = np.exp(1j * un * xn) * phi1 / (1j * un)
    F2 = np.exp(1j * un * xn) * phi2 / (1j * un)
    
    F1 = F1.real * hn
    F2 = F2.real * hn
    
    I1 = np.sum(F1, axis=-1) / np.pi
    I2 = np.sum(F2, axis=-1) / np.pi
    if isCall:
        P1 = 0.5 + I1
        P2 = 0.5 + I2
        res = S * P2 - np.exp(-r * tau) * K * P1
    else:
        P1 = 0.5 - I1
        P2 = 0.5 - I2
        res = np.exp(-r * tau) * K * P1 - S * P2
    return res


def heston_option_price_derivatives(S:Union[np.ndarray, float], K:Union[np.ndarray, float], tau:Union[np.ndarray, float], 
                            Nu:int, r:float, heston_params:np.ndarray, isCall=True) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return option price and it derivatives with respect to heston params

        S -- current stock price. if S is np.ndarray, S should have the same length as K

        K -- strikes.

        tau -- time to expiration. If tau is np.ndarray, tau should have the same length as K

        Nu -- number of points in fourier integral

        r, v0, theta, rho, k, sig -- model parameters

        isCall -- boolean flag.
        
        output shape
        res.shape == (len(K), )
        resDer.shape == (len(heston_params), len(K))
    '''   
    if not isinstance(K, np.ndarray):
        K = np.asarray([K])
    if isinstance(tau, np.ndarray):
        assert len(tau) == len(K)
    else:
        tau = np.asarray([tau])
    if isinstance(S, np.ndarray):
        assert len(S) == len(K)
    else:
        S = np.asarray([S])

    Nk = len(K)
    Nt = len(tau)
    Ndim = len(heston_params) // 5

    un, hn = getMesh(Nu)

    un = un.reshape(1, -1)
    hn = hn.reshape(1, -1)
    
    xn = np.log(S * np.exp(r * tau) / K).reshape(-1, 1)
    
    phi1 = np.ones((Nt, Nu), complex)
    phi2 = np.ones((Nt, Nu), complex)



    der1 = np.zeros( (Ndim * 5, Nt, Nu), complex )
    der2 = np.zeros( (Ndim * 5, Nt, Nu), complex )

    
    for i in range(Ndim):
        v, theta, rho, k, sig = heston_params[5 * i : 5 * i + 5]
        params = {"v0":v, "theta":theta, "rho":rho, "k":k, "sig":sig}

        _phi1, _der1 = phi_derivatives(un     , tau.reshape(-1, 1), **params)
        _phi2, _der2 = phi_derivatives(un - 1j, tau.reshape(-1, 1), **params)
        
        assert _phi1.shape == (Nt, Nu)
        assert _der1.shape == (5, Nt, Nu)

        phi1 = phi1 * _phi1
        phi2 = phi2 * _phi2
        
        der1[5 * i : 5 * i + 5, :] = _der1
        der2[5 * i : 5 * i + 5, :] = _der2

    F1 = np.exp(1j * un * xn) * phi1 / (1j * un) * hn
    F2 = np.exp(1j * un * xn) * phi2 / (1j * un) * hn

    I1 = np.sum(F1.real, axis=-1) / np.pi
    I2 = np.sum(F2.real, axis=-1) / np.pi

    IDer1 = np.sum( (F1 * der1).real, axis=-1 ) / np.pi
    IDer2 = np.sum( (F2 * der2).real, axis=-1 ) / np.pi
    assert IDer1.shape == (5 * Ndim, Nk)

    if isCall:
        P1 = 0.5 + I1
        P2 = 0.5 + I2
        res = S * P2 - np.exp(-r * tau) * K * P1
    else:
        P1 = 0.5 - I1
        P2 = 0.5 - I2
        res = np.exp(-r * tau) * K * P1 - S * P2
    
    resDer = S * IDer2 - np.exp(-r * tau) * K * IDer1 
    assert resDer.shape == (5 * Ndim, Nk)
    return res, resDer
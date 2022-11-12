from typing import Optional, Union, Tuple, List
import numpy as np

from nonlinear_optimization import nonlinear_optimization
from heston_option_price import heston_option_price, heston_option_price_derivatives
from heston import Heston


def gen_random_heston_params() -> np.ndarray:
    """
        This function gerenate random parameters for heston model

        Returns:
            heston_params(np.ndarray): generated heston params
    """
    eps = 1e-5

    v0 = np.random.rand(1) * 0.15 + 0.01
    theta = np.random.rand(1) * 0.15 + 0.01
    
    rho = -0.9 + (1.8) * np.random.rand(1)
    k = np.random.rand(1) * 2 + 1.0
    sig = np.random.rand(1) * 0.02 + 0.01
    
    return np.asarray([v0[0], theta[0], rho[0], k[0], sig[0]])


def proj_heston( heston_params : np.ndarray )->np.ndarray:
    """
        This funciton project heston parameters into valid range

        Attributes:
            heston_params(np.ndarray): model parameters
        
        Returns:
            heston_params(np.ndarray): clipped parameters
    """
    eps = 1e-3
    for i in range(len(heston_params) // 5):
        v0, theta, rho, k, sig = heston_params[i * 5 : i * 5 + 5]
        v0 = np.clip(v0, eps, 5.0)
        theta = np.clip(theta, eps, 5.0)
        rho = np.clip(rho, -1 + eps, 1 - eps)
        k = np.clip(k, eps, 10.0)
        sig = np.clip(sig, eps, 5.0)
        heston_params[i * 5 : i * 5 + 5] = v0, theta, rho, k, sig
    return heston_params


class HestonCalibrator:
    """ Class for heston model calibration and usage.
    
    Attributes:
        r(float): interest rate
        heston_params(np.ndarray): calibrated parameters of the heston model, heston_params = [v_0, theta, rho, k, sig]
        heston(heston): heston-object 
    """
        
    def __init__(self, interest_rate:float = 0, n_dim:int = 1, n_int:int = 200):
        """
            The __init__ method just save interest rate.
            
            Args:
                interest_rate(float): risk free interest rate, default value is zero
                n_dim(int): number of dimention in heston model, default is one
        """
        self.r = interest_rate
        self.n_dim = n_dim
        self.n_int = n_int

        self.heston_params = None
        self.heston = None

    
    def get_model(self)->Heston:
        """
            this funciton return fitted heston model
        """
        return self.heston


    def fit(self, 
            X: np.ndarray, 
            K: np.ndarray, 
            F: Union[float, np.ndarray], 
            T: Union[float, np.ndarray],
            typ: Union[bool, np.ndarray], 
            Niter:int=100, 
            weights:Optional[np.ndarray]=None, 
            heston_params:Optional[np.ndarray]=None) -> np.ndarray:
        """
            The fit method calibrate pararmeters of the heston model to market prices
            optimal params are saved in the heston_params attribute
            optimal model are saved in the heston attribute
            
            Args:
                X(np.ndarray): array of market prices
                K(np.ndarray): array of strikes
                F(float | np.ndarray): underlying futures price
                T(float, np.ndarray): expiration time
                typ(Union[bool, pn.ndarray]): type of option
                Niter(int): number of iteration
                weights(Optional[np.ndarray]): array of weights. If it's None, we use uniform weights
                heston_params(Optional[np.ndarray]): Initial model params.
                If it's None, we generate random initial parameters.
            Returns:
                fs(np.ndarray): array of errors on each iteration
        """
        if isinstance(F, float):
            F = np.ones_like(X) * F
        if isinstance(T, float):
            T = np.ones_like(X) * T
        if isinstance(typ, bool):
            typ = np.asarray( [typ] * len(X) )
        if weights is None:
            weights = np.ones_like(X)
        weights = weights / np.sum(weights)
        if heston_params is None:
            #generate random initial params
            heston_params = np.zeros( (5 * self.n_dim, ) )
            for i in range(self.n_dim):
                heston_params[i * 5: i * 5 + 5] = gen_random_heston_params()
        else:
            assert len(heston_params) == 5 * self.n_dim

        r = self.r
        n_int = self.n_int
        def get_residuals( heston_params:np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
            '''
                This function calculates residuals and Jacobian matrix
                Args:
                    heston_params(np.ndarray): model params
                Returns:
                    res(np.ndarray) : vector or residuals
                    J(np.ndarray)   : Jacobian
            '''
            C, J = heston_option_price_derivatives(F, K, T, n_int, r, heston_params)
            P = C + np.exp(-r * T) * ( K - F )
            X_ = C
            X_[~typ] = P[~typ]
            res = X_ - X
            return res * weights, J @ np.diag(weights)
        
        #optimization
        result = nonlinear_optimization(Niter, get_residuals, proj_heston, heston_params)
        self.heston_params = result['x']

        self.heston = Heston(self.heston_params, self.r, self.n_int)
        return result
        
        
    def predict(self, K: np.ndarray, 
                      F: Union[float, np.ndarray], 
                      T: Union[float, np.ndarray]) -> Tuple[ np.ndarray, np.ndarray ]:
        """
            The predict method returns option prices and implied volatility for given parameters K, F, T 
            
            Args:
                K(np.ndarray): array of strikes
                F(float | np.ndarray): underlying futures price
                T(float, np.ndarray): expiration time
            Returns:
                C(np.ndarray): option prices
                iv(np.ndarray) : implied volatility
        """
        assert not self.heston is None, "model is not fitted!"
        return self.heston(K, F, T)
        
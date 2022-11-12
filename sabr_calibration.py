from typing import Optional, Union, Tuple, List
import numpy as np

from nonlinear_optimization import nonlinear_optimization
from sabr_approx import sabr_approx, sabr_approx_derivatives
from sabr import SABR

def gen_random_sabr_params() -> np.ndarray:
    """
        This function gerenate random parameters for sabr model

        Returns:
            sabr_params(np.ndarray): generated sabr params
    """
    eps = 1e-5
    alpha = 0.3 * np.random.rand(1) + eps
    v = 1.0 * np.random.rand(1) + eps
    beta = 0.1 + 0.8 * np.random.rand(1)
    rho = -0.9 + (1.8) * np.random.rand(1)
    return np.asarray([alpha[0], v[0], beta[0], rho[0]] )


def proj_sabr( sabr_params : np.ndarray )->np.ndarray:
    """
        This funciton project sabr parameters into valid range

        Attributes:
            sabr_params(np.ndarray): model parameters
        
        Returns:
            sabr_params(np.ndarray): clipped parameters
    """
    alpha, v, beta, rho = sabr_params
    
    eps = 1e-6
    
    alpha = max(alpha, eps)
    v = max(v, eps)
    rho = np.clip(rho, -1 + eps, 1 - eps)
    beta = np.clip(beta, eps, 1 - eps)
    
    return np.asarray( [alpha, v, beta, rho] )


class SABRCalibrator:
    """ Class for SABR model calibration and usage.
    
    Attributes:
        r(float): interest rate
        sabr_params(np.ndarray): calibrated parameters of the SABR model, sabr_params = [sig_0, alpha, beta, rho]
        sabr(SABR): SABR-object 
    """
        
    def __init__(self, interest_rate:float = 0):
        """
            The __init__ method just save interest rate.
            
            Args:
                interest_rate(float): risk free interest rate, default value is zero
        """
        self.r = interest_rate
        self.sabr_params = None
        self.sabr = None

    
    def get_model(self)->SABR:
        """
            this funciton return fitter SABR model
        """
        return self.sabr
        
    
    def fit_iv(self, 
               iv0:np.ndarray, 
               K:np.ndarray, 
               F: Union[float, np.ndarray], 
               T: Union[float, np.ndarray], 
               Niter:int=100, 
               weights:Optional[np.ndarray]=None, 
               sabr_params:Optional[np.ndarray]=None,
               fit_beta:bool=True) -> np.ndarray:
        """
            The fit_iv method calibrate pararmeters of the SABR model to market implied volatility
            
            Args:
                iv0(np.ndarray): array of market implied volatility
                K(np.ndarray): array of strikes
                F(float | np.ndarray): underlying futures price
                T(float, np.ndarray): expiration time
                Niter(int): number of iteration
                weights(Optional[np.ndarray]): array of weights. If it's None, we use uniform weights
                sabr_params(Optional[np.ndarray]): Initial model params.
                If it's None, we generate random initial parameters.
                
            Returns:
                fs(np.ndarray): array of errors on each iteration
        """
        if isinstance(F, float):
            F = np.ones_like(iv0) * F
        if isinstance(T, float):
            T = np.ones_like(iv0) * T
        if weights is None:
            weights = np.ones_like(iv0)
        weights = weights / np.sum(weights)
        if sabr_params is None:
            sabr_params = gen_random_sabr_params()
        if not fit_beta:
            sabr_params[2] = 0.5
        def get_residals( sabr_params:np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
            '''
                This function calculates residuals and Jacobian matrix
                Args:
                    sabr_params(np.ndarray): model params
                Returns:
                    res(np.ndarray) : vector or residuals
                    J(np.ndarray)   : Jacobian
            '''
            alpha, v, beta, rho = sabr_params
            C, vega, iv, iv_alpha, iv_v, iv_beta, iv_rho = sabr_approx_derivatives(K, F, T, self.r, alpha, v, beta, rho)
            if not fit_beta:
                iv_beta *= 0
            res = iv - iv0
            J = np.asarray([iv_alpha, iv_v, iv_beta, iv_rho])
            return res * weights, J @ np.diag(weights)
        
        #optimization
        result = nonlinear_optimization(Niter, get_residals, proj_sabr, sabr_params)
        self.sabr_params = result['x']
        
        self.sabr = SABR(self.sabr_params, self.r)
        return result


    def fit_price(self, 
               X: np.ndarray, 
               K: np.ndarray, 
               F: Union[float, np.ndarray], 
               T: Union[float, np.ndarray],
               typ: Union[bool, np.ndarray], 
               Niter:int=100, 
               weights:Optional[np.ndarray]=None, 
               sabr_params:Optional[np.ndarray]=None,
               fit_beta: bool = True) -> np.ndarray:
        """
            The fit_iv method calibrate pararmeters of the SABR model to market prices
            
            Args:
                X(np.ndarray): array of market prices
                K(np.ndarray): array of strikes
                F(float | np.ndarray): underlying futures price
                T(float, np.ndarray): expiration time
                typ(Union[bool, pn.ndarray]): type of option
                Niter(int): number of iteration
                weights(Optional[np.ndarray]): array of weights. If it's None, we use uniform weights
                sabr_params(Optional[np.ndarray]): Initial model params.
                If it's None, we generate random initial parameters.
            Returns:
                fs(np.ndarray): array of errors on each iteration
        """
        if isinstance(F, float):
            F = np.ones_like(X) * F
        if isinstance(T, float):
            T = np.ones_like(X) * T
        if isinstance(typ, bool):
            typ = np.asarray( [typ] * len(F) )
        if weights is None:
            weights = np.ones_like(X)
        weights = weights / np.sum(weights)
        if sabr_params is None:
            sabr_params = gen_random_sabr_params()
        if not fit_beta:
            sabr_params[2] = 0.5
        def get_residals( sabr_params:np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
            '''
                This function calculates residuals and Jacobian matrix
                Args:
                    sabr_params(np.ndarray): model params
                Returns:
                    res(np.ndarray) : vector or residuals
                    J(np.ndarray)   : Jacobian
            '''
            
            alpha, v, beta, rho = sabr_params
            C, vega, iv, iv_alpha, iv_v, iv_beta, iv_rho = sabr_approx_derivatives(K, F, T, self.r, alpha, v, beta, rho)
            if not fit_beta:
                iv_beta *= 0.0
            P = C + np.exp(-self.r * T) * ( K - F )
            X_ = P
            X_[typ] = C[typ]
            res = X_ - X
            J = np.asarray([iv_alpha * vega, iv_v * vega, iv_beta * vega, iv_rho * vega])
            return res * weights, J @ np.diag(weights)
        
        #optimization
        result = nonlinear_optimization(Niter, get_residals, proj_sabr, sabr_params)
        self.sabr_params = result['x']
        
        self.sabr = SABR(self.sabr_params, self.r)
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
        assert not self.sabr is None, "model is not fitted!"
        return self.sabr(K, F, T)
        
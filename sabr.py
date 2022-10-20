from typing import Optional, Union, Tuple, List
import numpy as np

from sabr_approx import sabr_approx


class SABR:
    """ Class for SABR model
    
    Attributes:
        r(float): interest rate
        sabr_params(np.ndarray): parameters of the SABR model, sabr_params = [alpha, v, beta, rho]
    """
        
    def __init__(self, sabr_params:np.ndarray, interest_rate:float = 0):
        """
            The __init__ method just save interest rate.
            
            Args:
                interest_rate(float): risk free interest rate, default value is zero
        """
        self.sabr_params = sabr_params
        self.r = interest_rate
    
        
    def __call__(self, K: np.ndarray, 
                      F: Union[float, np.ndarray], 
                      T: Union[float, np.ndarray], is_call:bool = True) -> Tuple[ np.ndarray, np.ndarray ]:
        """
            This method returns option prices and implied volatility for given parameters K, F, T 
            
            Args:
                K(np.ndarray): array of strikes
                F(float | np.ndarray): underlying futures price
                T(float | np.ndarray): expiration time
                is_call(bool): is call 
            Returns:
                C(np.ndarray): option prices
                iv(np.ndarray) : implied volatility
        """
        C, iv = sabr_approx(K, F, T, self.r, *self.sabr_params) 
        P = C + np.exp(-self.r * T) * (K - F)
        X = C if is_call else P
        return X, iv
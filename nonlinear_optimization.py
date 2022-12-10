import numpy as np

from typing import Tuple, Callable, Dict

from tqdm import tqdm

def nonlinear_optimization(Niter:int, 
                          f:Callable[ [np.ndarray], Tuple[np.ndarray, np.ndarray]], 
                          proj:Callable[ [np.ndarray], np.ndarray ], 
                          x0:np.ndarray) -> Dict:
    ''' 
        Nonlinear least squares method, Levenberg-Marquardt Method
        
        Args:
            Niter(int): number of iteration
            f(Callable[ [np.ndarray], Tuple[np.ndarray, np.ndarray]]): 
                callable, gets vector of model parameters x as input, 
                returns tuple res, J, where res is numpy vector of residues, 
                J is jacobian of residues with respect to x 
            proj(Callable[ [np.ndarray], np.ndarray ]):
                callable, gets vector of model parameters x,
                returns vector of projected parameters 
            x0(np.ndarray): initial parameters

        Returns:
            result(dict): dictionary with results
            result['xs'] contains optimized parameters on each iteration
            result['objective'] contains norm of residuals on each iteration
            result['x'] is optimized parameters
    '''
    x = x0.copy()

    

    mu = 100.0
    nu1 = 2.0
    nu2 = 2.0

    fs = []
    res, J = f(x)
    F = np.linalg.norm(res)
    
    result = { "xs":[x], "objective":[F], "x":None }
    
    for i in range(Niter):
        I = np.diag(np.diag(J @ J.T)) + 1e-5 * np.eye(len(x))
        dx = np.linalg.solve( mu * I + J @ J.T, J @ res )
        x_ = proj(x - dx)
        res_, J_ = f(x_)
        F_ = np.linalg.norm(res_)
        if F_ < F:
            x, F, res, J = x_, F_, res_, J_
            mu /= nu1
            result['xs'].append(x)
            result['objective'].append(F)
        else:
            i -= 1
            mu *= nu2
            continue        
        eps = 1e-10
        if F < eps:
            break
        result['x'] = x
    return result
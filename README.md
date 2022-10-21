# CryptoOptionCalibration
Repo for Crypto Option Calibration project in CMF

Main result for sabr model is [here](https://github.com/dolmatovas/CryptoOptionCalibration/blob/main/sabr%20calibration.ipynb)

[Here](https://github.com/dolmatovas/CryptoOptionCalibration/blob/main/sabr%20calibration%20examples.ipynb) you can see examples of calibration of the SABR model to synthetic data.

Main result for heston model is [here](https://github.com/dolmatovas/CryptoOptionCalibration/blob/main/heston%20calibration.ipynb)

[Here](https://github.com/dolmatovas/CryptoOptionCalibration/blob/main/heston%20calibration%20examples.ipynb) you can see examples of calibration of the Heston model to synthetic data.

In sabr_approx.py the approximation formulas for implied volatility in the SABR model and its derivatives with respect to parameters are presented. 

In sabr.py the SABR object is defined. It's just a wrapper that stores model parameters and calls corresponding functions.

In sabr_calibration.py the SABRCalibrator object is defined. This object has methods fit_iv and fit_price. The first one fits the parameters of the SABR model to the market implied volatility, the second one fits them to the market option prices. 

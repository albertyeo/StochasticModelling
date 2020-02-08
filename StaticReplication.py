#Static Replication

'''
Evaluation of European derivatives using Black-Scholes model and Bachelier model
'''
#------------------------------------------------------------------------------
import pandas
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy import interpolate
import datetime as dt
#------------------------------------------------------------------------------
#Black-Scholes Model

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def BlackScholesDerivative(S, K, r, sigma, T):
    term1 = 10**(-8) * (S*np.exp(r*T+sigma**2*T))**3
    term2 = 0.5 * (np.log(S) + (r - 0.5*sigma**2)*T)
    term3 = 10.0
    return np.exp(-r*T) * (term1+term2+term3)
#------------------------------------------------------------------------------
#Bachelier Model
    
def BachelierDerivative(S, K, sigma, T):
    term1 = 10**(-8) * (S**3*(1+3*sigma**2*T))
    term2 = 0.5 * (np.log(S)-0.5*sigma**2*T)
    term3 = 10.0
    return term1+term2+term3
#------------------------------------------------------------------------------
#Implied Call & Put Volatility
    
def impliedCallVolatility(S, K, r, price, T):
    impliedVol = brentq(lambda x: price -
                        BlackScholesCall(S, K, r, x, T),
                        1e-6, 1)
    return impliedVol

def impliedPutVolatility(S, K, r, price, T):
    impliedVol = brentq(lambda x: price -
                        BlackScholesPut(S, K, r, x, T),
                        1e-6, 1)
    return impliedVol
#------------------------------------------------------------------------------
#DataFrame
    
google_df = pandas.read_csv('GOOG.csv')
rate_df = pandas.read_csv('discount.csv')
call_df = pandas.read_csv("goog_call.csv")
call_df['mid_price'] = (call_df['best_bid'] + call_df['best_offer'])/2
put_df = pandas.read_csv("goog_put.csv")
put_df['mid_price'] = (put_df['best_bid'] + put_df['best_offer'])/2
#------------------------------------------------------------------------------
#Interpolation
x = rate_df['Day']
y = rate_df['Rate (%)']
f = interpolate.interp1d(x,y)
#------------------------------------------------------------------------------
#Parameters

n = len(call_df.index)
strike = call_df['strike'].values

today = dt.date(2013, 8, 30)
expiry = dt.date(2015, 1, 17)

T = (expiry-today).days/365.0
S = 846.9
r = f(T*365)/100
F = np.exp(r*T)*S

K = 850
atm_call = (100+102.8)/2
atm_put = (101.8+104)/2
sigma_call = impliedCallVolatility(S, K, r, atm_call, T)
sigma_put = impliedPutVolatility(S, K, r, atm_put, T)
sigma = (sigma_call + sigma_put)/2
print(sigma)
#------------------------------------------------------------------------------
#Pricing
price_blackscholes = BlackScholesDerivative(S, K, r, sigma, T)
print('---------------------Black-Scholes Model---------------------')
print('The price of the derivative contract is: ', price_blackscholes)

print('\n')

price_bachelier = BachelierDerivative(S, K, sigma, T)
print('-----------------------Bachelier Model-----------------------')
print('The price of the derivative contract is: ', price_bachelier)

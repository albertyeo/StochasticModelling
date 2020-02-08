#Model Calibration

'''
Calibration of displaced-diffusion model and SABR Model (beta = 0.8) to match
the option prices based on Google stock
'''
#------------------------------------------------------------------------------
import pandas
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy import interpolate
import matplotlib.pylab as plt
import datetime as dt
from scipy.optimize import least_squares
from math import log, sqrt, exp
from scipy.optimize import fsolve
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
#------------------------------------------------------------------------------
#Displaced-Diffusion Model

def DisplacedDiffusionCall(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F/beta*norm.cdf(c1) - ((1-beta)/beta*F + K)*norm.cdf(c2))

def DisplacedDiffusionPut(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(((1-beta)/beta*F + K)*norm.cdf(-c2) - F/beta*norm.cdf(-c1))
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

beta1 = 0.2
beta2 = 0.4
beta3 = 0.6
beta4 = 0.8
#------------------------------------------------------------------------------
#Displaced-Diffusion Graph

summary = []
for i in range(n):
    K = strike[i]
    if K <= 850:
        impliedvol_market = impliedPutVolatility(S, K, r, put_df['mid_price'][i], T)

        price_lognormal = BlackScholesPut(S, K, r, sigma, T)
        impliedvol_lognormal = impliedPutVolatility(S, K, r, price_lognormal, T)
        
        price_dd1 = DisplacedDiffusionPut(S, K, r, sigma, T, beta1)
        impliedvol_dd1 = impliedPutVolatility(S, K, r, price_dd1, T)
        
        price_dd2 = DisplacedDiffusionPut(S, K, r, sigma, T, beta2)
        impliedvol_dd2 = impliedPutVolatility(S, K, r, price_dd2, T)
        
        price_dd3 = DisplacedDiffusionPut(S, K, r, sigma, T, beta3)
        impliedvol_dd3 = impliedPutVolatility(S, K, r, price_dd3, T)
        
        price_dd4 = DisplacedDiffusionPut(S, K, r, sigma, T, beta4)
        impliedvol_dd4 = impliedPutVolatility(S, K, r, price_dd4, T)
        
    elif K > 850:
        impliedvol_market = impliedCallVolatility(S, K, r, call_df['mid_price'][i], T)

        price_lognormal = BlackScholesCall(S, K, r, sigma, T)
        impliedvol_lognormal = impliedCallVolatility(S, K, r, price_lognormal, T)
        
        price_dd1 = DisplacedDiffusionCall(S, K, r, sigma, T, beta1)
        impliedvol_dd1 = impliedCallVolatility(S, K, r, price_dd1, T)
        
        price_dd2 = DisplacedDiffusionCall(S, K, r, sigma, T, beta2)
        impliedvol_dd2 = impliedCallVolatility(S, K, r, price_dd2, T)
        
        price_dd3 = DisplacedDiffusionCall(S, K, r, sigma, T, beta3)
        impliedvol_dd3 = impliedCallVolatility(S, K, r, price_dd3, T)
        
        price_dd4 = DisplacedDiffusionCall(S, K, r, sigma, T, beta4)
        impliedvol_dd4 = impliedCallVolatility(S, K, r, price_dd4, T)
        
    summary.append([K, 
                    impliedvol_market, 
                    impliedvol_lognormal, 
                    impliedvol_dd1,
                    impliedvol_dd2,
                    impliedvol_dd3,
                    impliedvol_dd4
                    ])
    
df = pandas.DataFrame(summary, columns=['strike', 
                                        'impliedvol_market', 
                                        'impliedvol_lognormal', 
                                        'impliedvol_dd1',
                                        'impliedvol_dd2',
                                        'impliedvol_dd3',
                                        'impliedvol_dd4'
                                        ])
#------------------------------------------------------------------------------
#Displaced-Diffusion

def ddcalibration(x, S, strikes, r, vols, T):
    err = 0.0
    for i, vol in enumerate(vols):
        price = DisplacedDiffusionCall(S, strikes[i], r, sigma, T, x[0])
        err += (vol - impliedCallVolatility(S, strikes[i], r, price, T))**2
        
    return err

initialGuess = [0.3]
res = least_squares(lambda x: ddcalibration(x, 
                                            S, 
                                            df['strike'].values,
                                            r,
                                            df['impliedvol_market'],
                                            T),
                    initialGuess)
        
beta_dd = res.x[0]
#------------------------------------------------------------------------------
summary_dd = []
for i in range(n):
    K = strike[i]
    if K <= 850:        
        price_dd5 = DisplacedDiffusionPut(S, K, r, sigma, T, beta_dd)
        impliedvol_dd5 = impliedPutVolatility(S, K, r, price_dd5, T)
        
    elif K > 850:        
        price_dd5 = DisplacedDiffusionCall(S, K, r, sigma, T, beta_dd)
        impliedvol_dd5 = impliedCallVolatility(S, K, r, price_dd5, T)   
    summary_dd.append([impliedvol_dd5])

df['impliedvol_dd5'] = np.array(summary_dd)

plt.plot(df['strike'],df['impliedvol_market'], 'go', label = 'Market')
plt.plot(df['strike'],df['impliedvol_dd5'],'m:', label = 'Displaced Diffusion Model')

plt.legend(loc="upper right")
plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.show()
print('-----------Displaced-Diffusion Model------------')
print('The value that most closely match the market is: ')
print('Beta = ', beta_dd)

plt.plot(df['strike'],df['impliedvol_market'], 'go', label = 'Market')
plt.plot(df['strike'],df['impliedvol_lognormal'], label = 'Lognormal Model')
plt.plot(df['strike'],df['impliedvol_dd1'],'r:', label = '\u03B2 = 0.2')
plt.plot(df['strike'],df['impliedvol_dd2'],'b:', label = '\u03B2 = 0.4')
plt.plot(df['strike'],df['impliedvol_dd3'],'y:', label = '\u03B2 = 0.6')
plt.plot(df['strike'],df['impliedvol_dd4'],'g:', label = '\u03B2 = 0.8')

plt.legend(loc="upper right")
plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.show()
#------------------------------------------------------------------------------
#SABR

beta_SABR = 0.8

rho1 = -0.5
rho2 = 0.0
rho3 = 0.5

nu1 = 0.1
nu2 = 0.3
nu3 = 0.5


def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*log(F/X)
        zhi = log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma

def sabrcalibration(x, strikes, vols, F, T, beta):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], beta, x[1], x[2]))**2

    return err

initialGuess2 = [0.02, 0.2, 0.1]
res_SABR = least_squares(lambda x: sabrcalibration(x,
                                              df['strike'].values,
                                              df['impliedvol_market'].values,
                                              F,
                                              T,
                                              beta_SABR),
                    initialGuess2)
alpha = res_SABR.x[0]
rho = res_SABR.x[1]
nu = res_SABR.x[2]
#------------------------------------------------------------------------------
#SABR Graph

summary2 = []
for i in range(n):
    K = strike[i]
    if K <= 850:
        impliedvol_market = impliedPutVolatility(S, K, r, put_df['mid_price'][i], T)

        price_lognormal = BlackScholesPut(S, K, r, sigma, T)
        impliedvol_lognormal = impliedPutVolatility(S, K, r, price_lognormal, T)
        
    elif K > 850:
        impliedvol_market = impliedCallVolatility(S, K, r, call_df['mid_price'][i], T)

        price_lognormal = BlackScholesCall(S, K, r, sigma, T)
        impliedvol_lognormal = impliedCallVolatility(S, K, r, price_lognormal, T)
        
    impliedvol_SABR = SABR(F, K, T, alpha, beta_SABR, rho, nu)
    
    impliedvol_rho1 = SABR(F, K, T, alpha, beta_SABR, rho1, nu)
    impliedvol_rho2 = SABR(F, K, T, alpha, beta_SABR, rho2, nu)
    impliedvol_rho3 = SABR(F, K, T, alpha, beta_SABR, rho3, nu)

    impliedvol_nu1 = SABR(F, K, T, alpha, beta_SABR, rho, nu1)
    impliedvol_nu2 = SABR(F, K, T, alpha, beta_SABR, rho, nu2)
    impliedvol_nu3 = SABR(F, K, T, alpha, beta_SABR, rho, nu3)

    summary2.append([K,
                    impliedvol_market,
                    impliedvol_lognormal,
                    impliedvol_SABR, 
                    impliedvol_rho1, 
                    impliedvol_rho2,
                    impliedvol_rho3,
                    impliedvol_nu1, 
                    impliedvol_nu2,
                    impliedvol_nu3
                    ])
    
df2 = pandas.DataFrame(summary2, columns=['strike', 
                                          'impliedvol_market',
                                          'impliedvol_lognormal',
                                          'impliedvol_SABR', 
                                          'impliedvol_rho1', 
                                          'impliedvol_rho2',
                                          'impliedvol_rho3', 
                                          'impliedvol_nu1', 
                                          'impliedvol_nu2',
                                          'impliedvol_nu3'
                                          ])


plt.plot(df2['strike'],df2['impliedvol_market'], 'go', label = 'Market')
plt.plot(df2['strike'],df2['impliedvol_SABR'],'c:', label = 'SABR Model')
plt.legend(loc="upper right")
plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.show()

print('--------------------SABR Model--------------------')
print('The values that most closely match the market are: ')
print('Alpha = ', alpha)
print('Beta = ', beta_SABR)
print('Rho = ', rho)
print('Nu = ', nu)

plt.plot(df2['strike'],df2['impliedvol_market'], 'go', label = 'Market')
plt.plot(df2['strike'],df2['impliedvol_lognormal'], label = 'Lognormal Model')
plt.plot(df2['strike'],df2['impliedvol_rho1'],'r:', label = '\u03C1 = -0.5')
plt.plot(df2['strike'],df2['impliedvol_rho2'],'b:', label = '\u03C1 = 0.0')
plt.plot(df2['strike'],df2['impliedvol_rho3'],'y:', label = '\u03C1 = 0.5')

plt.legend(loc="upper right")
plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.show()

plt.plot(df2['strike'],df2['impliedvol_market'], 'go', label = 'Market')
plt.plot(df2['strike'],df2['impliedvol_lognormal'], label = 'Lognormal Model')
plt.plot(df2['strike'],df2['impliedvol_nu1'],'r:', label = '\u03BD = 0.1')
plt.plot(df2['strike'],df2['impliedvol_nu2'],'b:', label = '\u03BD = 0.3')
plt.plot(df2['strike'],df2['impliedvol_nu3'],'y:', label = '\u03BD = 0.5')

plt.legend(loc="upper right")
plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.show()

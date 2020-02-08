#Analytical Option Formulae

'''
Calculation of the value of the following European options:
- Vanilla call/put
- Digital cash-or-nothing call/put
- Digital asset-or-nothing call/put
based on the following models:
1. Black-Scholes model
2. Bachelier model
3. Black76 model
4. Displaced-diffusion model
'''
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

def BlackScholesDCashCall(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(d2)

def BlackScholesDCashPut(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(-d2)

def BlackScholesDAssetCall(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1)

def BlackScholesDAssetPut(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(-d1)
#------------------------------------------------------------------------------
#Bachelier Model
    
def BachelierCall(S, K, sigma, T):
    c = (S-K) / (sigma*S*np.sqrt(T))
    return (S-K)*norm.cdf(c) + sigma*S*np.sqrt(T)*norm.pdf(c)

def BachelierPut(S, K, sigma, T):
    c = (S-K) / (sigma*S*np.sqrt(T))
    return (K-S)*norm.cdf(-c) + sigma*S*np.sqrt(T)*norm.pdf(-c)   

def BachelierDCashCall(S, K, sigma, T):
    c = (S-K) / (sigma*S*np.sqrt(T))
    return norm.cdf(c)

def BachelierDCashPut(S, K, sigma, T):
    c = (S-K) / (sigma*S*np.sqrt(T))
    return norm.cdf(-c)

def BachelierDAssetCall(S, K, sigma, T):
    c = (S-K) / (sigma*S*np.sqrt(T))
    return S*norm.cdf(c) + sigma*S*np.sqrt(T)*norm.pdf(c)

def BachelierDAssetPut(S, K, sigma, T):
    c = (S-K) / (sigma*S*np.sqrt(T))
    return S*norm.cdf(-c) - sigma*S*np.sqrt(T)*norm.pdf(-c)
#------------------------------------------------------------------------------
#Black76 Model

def Black76Call(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F*norm.cdf(c1) - K*norm.cdf(c2))

def Black76Put(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(K*norm.cdf(-c2) - F*norm.cdf(-c1))

def Black76DCashCall(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(c2)

def Black76DCashPut(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(-c2)

def Black76DAssetCall(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    #c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return F*disc*norm.cdf(c1)

def Black76DAssetPut(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    #c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return F*disc*norm.cdf(-c1)
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

def DisplacedDiffusionDCashCall(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(c2)

def DisplacedDiffusionDCashPut(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(-c2)

def DisplacedDiffusionDAssetCall(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F/beta*norm.cdf(c1) - ((1-beta)/beta*F)*norm.cdf(c2))

def DisplacedDiffusionDAssetPut(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F/beta*norm.cdf(-c1) - ((1-beta)/beta*F)*norm.cdf(-c2))

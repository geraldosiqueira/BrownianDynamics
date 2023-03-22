import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numba import jit

T = 5
ni = 1
k=1

# Função da dinâmica de Langevin
@jit
def Langevin(N, Nt, dt, k, T):
    sigma = np.sqrt(2*k*T*dt/ni)
    x = np.zeros( [N, Nt] )
    y = np.zeros( [N, Nt] )
    for i in range(1,Nt):
            ux = np.random.normal(0,1,N)
            x[:, i] = x[:, i-1] + sigma*ux[:]
            uy = np.random.normal(0,1,N)
            y[:, i] = y[:, i-1] + sigma*uy[:]

    return x, y

N, tempo, dt = 10000, 50, 0.01
T = [0.1, 0.5, 1, 1.5, 2]
tzinho = [1, 2, 5, 10, 20, 40]
fig, ax = plt.subplots(2, len(T), figsize = (8,12))
for m in range(len(T)):
    x, y = Langevin(N, tempo, dt, k, T[m])
    x_gr , y_gr = [], []
    for k in tzinho:
        x_gr.append( np.mean((x[:,k])**2) - np.mean((x[:,k]))**2)
        y_gr.append( np.mean((y[:,k])**2) - np.mean((y[:,k]))**2)
    ax[0,m].plot(tzinho, x_gr, '.')
    ax[1,m].plot(tzinho, y_gr, '.')

    X = tzinho
    Y = x_gr
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    cof_angx = results.params[1]
    cof_linx = results.params[0]
    ax[0,m].plot(tzinho, [cof_angx*i + cof_linx for i in tzinho], color='green')
    print(cof_angx)

    X = tzinho
    Y = y_gr
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    cof_angy = results.params[1]
    cof_liny = results.params[0]
    ax[1,m].plot(tzinho, [cof_angy*i + cof_liny for i in tzinho], color = 'blue')  
fig.suptitle("Desvio médio quadrático pelo tempo para diferentes temperaturas")
ax[0,0].set_title(" Δx² vs. t")
ax[1,0].set_title(" Δy² vs. t")
plt.show()

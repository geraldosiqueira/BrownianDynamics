import numpy as np
import matplotlib.pyplot as plt
from numba import jit

T = 5
ni = 1
k=1

@jit
def Histograma(x, t, Nhist, a, b):
    x_aux = []
    for k in range( np.size(x[:,t])):
        if x[k,t] >= a and x[k,t] <= b:
            x_aux.append(x[k,t])
        
    dx = (b-a)/Nhist
    Xh = np.arange(a, b, dx)
    Hist = np.zeros(Nhist)
    N = np.size(x_aux)
    for i in range(N):
        j = int((x_aux[i] - a)/dx)
        if j<Nhist and j>0:
            Hist[j] += 1
    Hist = Hist/(N*dx)
    return Xh, Hist

@jit
def Langevin(N, tempo, dt, k, T):
    Nt = int(tempo/dt)
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
T=5
x, y = Langevin(N, tempo, dt, k, T)
fig, ax = plt.subplots(2)
for tempos in [1, 2, 5, 10, 20, 40]:
    Xh, histx = Histograma(x, tempos, 100, -7, 7)
    ax[0].plot(Xh, histx, label=f"t={tempos}")
    ax[0].legend()
    Yh, histy = Histograma(y, tempos, 100, -7, 7)
    ax[1].plot(Yh, histy, label=f"t={tempos}")
    ax[1].legend()
plt.show()
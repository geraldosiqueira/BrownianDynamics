# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:46:52 2022

@author: Geraldo Siqueira
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

N, tempo, dt, k = 10000, 5, 0.01, 1
T = [0.1, 0.5, 1, 1.5, 2]
ni = 1
Nt = int(tempo/dt)
ux = np.random.normal(0,1)
uy = np.random.normal(0,1)
x0 = 0
y0 = 0

@jit
def Histograma(x, t, Nhist, a, b):
    x_aux = []
    for k in range( np.size(x[t,:])):
        if x[t,k] >= a and x[t,k] <= b:
            x_aux.append(x[t,k])
        
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

# Função da dinâmica de Langevin
@jit
def Langevin(N, Nt, dt, k, T):
    sigma = np.sqrt(2*k*T*dt/ni)
    x = np.zeros( [Nt, N] )
    y = np.zeros( [Nt, N] )
    x[:, 0] = x0
    y[:, 0] = y0
    for i in range(1,Nt):
            ux = np.random.normal(0,1,N)
            x[i, :] = x[i-1, :] + sigma*ux[:]
            uy = np.random.normal(0,1,N)
            x[i, :] = y[i-1, :] + sigma*uy[:]

    return x, y

fig, ax = plt.subplots(2)
for j in range(len(T)):
    x, y = Langevin(N, Nt, dt, k, T[j])
    Xh, histx = Histograma(x, tempo, 40, -2, 2)
    ax[0].plot(Xh, histx, label=f"T={T[j]}")
    ax[0].legend()
    Yh, histy = Histograma(x, tempo, 40, -2, 2)
    ax[1].plot(Yh, histy, label=f"T={T[j]}")
    ax[1].legend()
plt.show()


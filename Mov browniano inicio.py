# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 08:20:18 2022

@author: Geraldo Siqueira
"""

import numpy as np
import matplotlib.pyplot as plt

N, tempo, dt, k, T = 1, 5, 0.01, 1, 2
ni = 1
Nt = int(tempo/dt)
ux = np.random.normal(0,1)
uy = np.random.normal(0,1)
x0 = 0
y0 = 0
sigma = np.sqrt(2*k*T*dt/ni)

def movX(x, ux):
    return x + sigma*ux

def movY(y, uy):
    return y + sigma*uy

# Função da dinâmica de Langevin
def Langevin(N, Nt, dt, k, T):
    x = np.zeros( [Nt, N] )
    y = np.zeros( [Nt, N] )
    x[:, 0] = x0
    y[:, 0] = y0
    for n in range(1,Nt):
        for i in range(N):
            ux = np.random.normal(0,1)
            x[n, i] = movX(x[n-1, i], ux)
            uy = np.random.normal(0,1)
            y[n, i] = movY(y[n-1, i], uy)

    return x, y

x, y = Langevin(N, Nt, dt, k, T)

plt.plot(x,y)
plt.title("Uma partícula com T = 2")
plt.xlabel("x")
plt.ylabel("y")
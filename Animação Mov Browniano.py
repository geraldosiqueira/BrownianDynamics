# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:24:30 2022

@author: Geraldo Siqueira
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit

k=1
ni = 1

def U(x,y):
    return 0.5*k*(x**2 + y**2)

# Função da dinâmica de Langevin
@jit
def Langevin(N, Nt, dt, k, T):
    sigma = np.sqrt(2*k*T*dt/ni)
    x = np.zeros( [N, Nt] )
    y = np.zeros( [N, Nt] )
    x[:,0] = 1
    y[:,0] = 0
    for i in range(1,Nt):
            ux = np.random.normal(0,1,N)
            x[:, i] = x[:, i-1] - (1/ni)*k*x[:, i-1]*dt+ sigma*ux[:]
            uy = np.random.normal(0,1,N)
            y[:, i] = y[:, i-1] - (1/ni)*k*y[:, i-1]*dt + sigma*uy[:]

    return x, y

N, tempo, dt= 1000, 5, 0.01
Nt = int(tempo/dt)
T = 0.001
x, y = Langevin(N, Nt, dt, k, T)

#Configurando a figura onde será feita a animação
L = 2.0
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
ax.set_aspect('equal')

# Objetos que receberão os dados (configurações) e texto a serem mostrados em cada quadro
confs, = ax.plot([], [], 'ko', ms=5, alpha=0.1)
texto = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# Converte os dados recebidos em cada iteração em objetos a serem mostrados na figura
def animate(n):
    confs.set_data(x[:,n], y[:,n])
    texto.set_text('Tempo = %.2f' % (n*dt))
    return confs, texto

# Constrói a animação
ani = FuncAnimation(fig, animate, frames=Nt,interval=50)


plt.show()



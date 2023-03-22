import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit
import matplotlib.animation as animation

k=1
ni = 1
kb = 1

def U(x,y):
    return 0.5*k*(x**2 + y**2)

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

N, tempo, dt = 1000, 5, 0.01
Nt = int(tempo/dt)
T = 0.1
A = np.sqrt(k/(2*np.pi*kb*T))
x, y = Langevin(N, Nt, dt, k, T)

L = 1.5
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(0, 5))

# Objetos que receberão os dados (configurações) e texto a serem mostrados em cada quadro
histo, = ax.plot([], [], 'ko', ms=5)
texto = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# Distribuição de equilíbrio
X = np.arange(-L,L,0.01)
P = A*np.exp(-k*(X**2)/(2*kb*T))
Boltz, = ax.plot([], [], 'r-', lw=2)

# Converte os dados recebidos em cada iteração em objetos a serem mostrados na figura
def animate(n):
    Xh, hist = Histograma(x,n,50,-2,2)
    Boltz.set_data(X, P)
    histo.set_data(Xh,hist)
    texto.set_text('Tempo = %.2f' % (n*dt))
    return histo, texto, Boltz

# Constrói a animação
ani = FuncAnimation(fig, animate, frames=Nt, interval=50)

plt.show()
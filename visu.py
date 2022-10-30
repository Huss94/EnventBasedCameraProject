import scipy.io
import numpy as np 
import scipy.linalg
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
mat = scipy.io.loadmat('s.mat')

x = mat['x']
y = mat['y']
p = mat['p']

#ON convertie le temps en miliseconde
ts = mat['ts']


#Alias : 
arr = np.array

def cond(Mat):
    vp = np.linalg.eigvals(Mat)
    return np.max(vp)/np.min(vp)

def isMatInversible(A):
    #On utilisera plutot, la fonction de conditionnlement de np array plutot que la notre ci dessus
    # On aurait put utiliser notre fonction de conditionnemnet qui donne des resultat quasi similaire, 
    # Je prefere cependant utilsier la bibliotheque numpy

    return np.linalg.cond(A, np.inf) < 100

def trace_plan(x,y,t, A,B, tcoeff= 0):
    X,Y = np.meshgrid(np.linspace(min(x) - 1, max(x) +1,10), np.linspace(min(y) - 1, max(y) + 1, 10))
    Z = A*X + B*Y + tcoeff # Le coefficient de t est 1
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, t, cmap="greens")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2) 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('t')
    plt.show()
    print(len(x))

    

def cramer_solve(A, B):
    A = np.array(A)
    B = np.array(B)

    assert A.shape[0] == A.shape[1]
    assert len(B.shape) == 1

    detA = np.linalg.det(A)

    sol = []
    for i in range(A.shape[1]):
        Ai = A.copy()
        Ai[:,i] = B
        detAi = np.linalg.det(Ai)
        sol.append(detAi/detA)
    return sol

def compute_lmsq2(x,y,ts,show_plane = False):
    A = np.c_[x, y, np.ones(x.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, ts)    # coefficients
    if show_plane:
        trace_plan(x,y,ts, *C)
    return C


def compute_lmsq(x,y,ts, show_plane = False):
    """
    On cherche un plan tel que : Ax + By + t + C = 0
    
    cette fonction retourne les paramates A,B,C du plan qui fit les données
    """
    if len(x) < 4:
        return None

    xmean = np.mean(x)
    ymean = np.mean(y)
    tsmean = np.mean(ts)
    x_tilde = x - xmean 
    y_tilde = y - ymean
    ts_tilde = ts - tsmean


    sumxy = np.sum(x_tilde*y_tilde)
    sumx_ts = np.sum(x_tilde*ts_tilde)
    sumy_ts = np.sum(y_tilde*ts_tilde)
    T = [sumx_ts, sumy_ts]


    M = [[np.sum(x_tilde**2), sumxy], [sumxy, np.sum(y_tilde**2)]]
    # Pour résoudre le systeme, on va utiliser la regle de cramer comme indiquée dans le papier, pour cela il faut que la matrice soit inversible
    # Pour cel a on calcul le conditionnement de la matrice M, si le conditionnemtn de la matrice est ttrop grand, cela peut ammener a des instabilité
    # Le conditionnelmtn ocrrespond a la valeur propre maximum sur la valeur prpme la plus petite

    
    # Si c'est inversible, on résoud avec le systeme de cramer avec notre fonction cramer_solve
    if isMatInversible(M):
        # A, B= np.linalg.solve(M,T)

        # https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
        A,B = cramer_solve(M, T)
        C = tsmean - A*xmean - B*ymean 
        if show_plane: 
            trace_plan(x, y, ts, A,B,C)
        return A,B,C
    return None


def compute_velocity(a,b):
    return -np.array([a,b])/(a**2 + b **2) 






def local_plane_fitting(x, y , p ,ts, L, dt):
    th1 = 1e-5
    th2 = 0.05
    v = np.zeros((len(x), 2))
    for i in tqdm(range(len(x))): 

        nei = np.where((x[i] - L <= x ) & (x<= x[i] +L) &
                        (y[i] - L<= y ) & (y<= y[i] +L) &
                        (ts[i] - dt <= ts) & (ts <= ts[i] + dt))[0]
        
        #Compute lmsq calcul les coefficient du plan. Retourne none 
        Pi0 = compute_lmsq(x[nei], y[nei], ts[nei])
        
        if Pi0 is None:
            continue

        eps = 10e6
        while eps > th1:
            A,B,C = Pi0
            indices_to_delete = np.argwhere(A*x[nei] + B*y[nei] + C - ts[nei] > th2)[:,0]
            n_nei = len(nei)     
            nei = np.delete(nei, indices_to_delete)
            if len(nei) == n_nei:
                #Dans ce cas la on a enelvé aucun event, alors on considère directement Pi0 comme le plan solution
                break            

            Pi = compute_lmsq(x[nei], y[nei], ts[nei]) 
            if Pi is not None:
                eps = np.linalg.norm(arr(Pi) - arr(Pi0))
                Pi0 = Pi
            else: 
                break
        v[i] = compute_velocity(Pi0[0], Pi0[1])
    return v





def compute_local_flow(x, y , p , ts, N_spatial = 5, N_temporel = 100):
    # Boucle a travers tous les evements
    for i in range(len(x)):
        neighbors = np.where((x[i] - N_spatial <= x ) & (x<= x[i] +N_spatial) &
                        (y[i] - N_spatial <= y ) & (y<= y[i] +N_spatial) &
                        (ts[i] - N_temporel <= ts) & (ts <= ts[i] + N_temporel)
        )
    return neighbors




v = local_plane_fitting(x,y,p,ts,5, 100)
# s = compute_local_flow(x,y,p,ts)
# print(len(s[0]))

data = np.load("test.npy")

# C = compute_lmsq(data[:,0], data[:,1], data[:,2], True)
# print(C)
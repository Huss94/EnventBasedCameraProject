import scipy.io
import numpy as np
import scipy.linalg
import time
from time import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
mat = scipy.io.loadmat('s.mat')

x = mat['x']
y = mat['y']
p = mat['p']

# ON convertie le temps en miliseconde
ts = mat['ts']





# Alias :
arr = np.array


def cond(Mat):
    # Retourne le conditionnement de la matrice
    vp = np.linalg.eigvals(Mat)
    return np.max(vp)/np.min(vp)


def isMatInversible(A):
    # On utilisera plutot, la fonction de conditionnlement de np array plutot que la notre ci dessus
    # On aurait put utiliser notre fonction de conditionnemnet qui donne des resultat quasi similaire,
    return np.linalg.cond(A, np.inf) < 100


def trace_plan(x, y, t, A, B, tcoeff=0):
    X, Y = np.meshgrid(np.linspace(min(x) - 1, max(x) + 1, 10),
                       np.linspace(min(y) - 1, max(y) + 1, 10))
    Z = A*X + B*Y + tcoeff  # Le coefficient de t est 1
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
        Ai[:, i] = B
        detAi = np.linalg.det(Ai)
        sol.append(detAi/detA)
    return sol


def compute_lmsq2(x, y, ts, show_plane=False):
    A = np.c_[x, y, np.ones(x.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, ts)    # coefficients
    if show_plane:
        trace_plan(x, y, ts, *C)
    return C


def compute_lmsq(x, y, ts, show_plane=False):
    """
    On cherche un plan tel que : Ax + By + t + C = 0

    cette fonction retourne les paramates A,B,C du plan qui fit les données
    retouren None dnas le cas où on ne trouve aucun plan
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
        A, B = cramer_solve(M, T)
        C = tsmean - A*xmean - B*ymean
        if show_plane:
            trace_plan(x, y, ts, A, B, C)
        return A, B, C
    return None


def compute_velocity(a, b):
    return -np.array([a, b])/(a**2 + b ** 2)


def get_circular_coordinates(x, y, R):
    """
    Retourne les coordonnées circulaires sur un rayon R autour de x, y
    """
    teta = np.linspace(-np.pi, np.pi, 2*R)
    x_coord = x + (R*np.cos(teta)).astype(int)
    y_coord = y + (R*np.sin(teta)).astype(int)

    coord = arr([x_coord, y_coord]).T

    return np.unique(coord, axis=0)


def local_plane_fitting(x, y, ts, nei):
    th1 = 1e-5
    th2 = 0.05

    Pi0 = compute_lmsq(x[nei], y[nei], ts[nei])

    if Pi0 is None:
        return None

    eps = 10e6
    while eps > th1:
        A, B, C = Pi0
        indices_to_delete = np.argwhere(
            A*x[nei] + B*y[nei] + C - ts[nei] > th2)[:, 0]
        n_nei = len(nei)
        nei = np.delete(nei, indices_to_delete)
        if len(nei) == n_nei:
            # Dans ce cas la on a enelvé aucun event, alors on considère directement Pi0 comme le plan solution
            break

        Pi = compute_lmsq(x[nei], y[nei], ts[nei])
        if Pi is not None:
            eps = np.linalg.norm(arr(Pi) - arr(Pi0))
            Pi0 = Pi
        else:
            break
    return Pi0


def compute_local_flow(x, y, p, ts, N=5, dt=1000):
    Un_tab = np.zeros((len(x),2))
    vx_tab = np.zeros(len(x))
    vy_tab = np.zeros(len(x))

    # Boucle a travers tous les evements
    for e in tqdm(range(len(x))):

        # COMPUTE LOCAL FLOW
        nei2 = np.where((x[e] - N <= x ) & (x<= x[e] +N) &
                        (y[e] - N <= y ) & (y<= y[e] +N) &
                        (ts[e] - dt <= ts) & (ts <= ts[e] + dt)
        )[0]

        # La facon de faire plus haut est plus intuitive mais plus energivore
        nei = np.where((ts[e] - dt <= ts) & (ts <= ts[e] + dt))[0]
        nei = nei[np.where((x[e] - N <= x[nei]) & (x[nei] <= x[e] + N))[0]]
        nei = nei[np.where((y[e] - N <= y[nei]) & (y[nei] <= y[e] + N))[0]]

        # le plan ax + by + t + c = 0
        P = local_plane_fitting(x, y, ts, nei)
        if P is None:
            continue
        a, b, c = P

        Inliners_count = 0
        # arr est un alias pour np array
        z_chap = np.sqrt(a**2 + b**2)

        # En annalysant le code en C++ de l'auteur, on l'a vu définir uchap de la sorte
        # U_chap = z_chap
        U_chap = 1 / z_chap

        for i in range(len(x[nei])):
            t_chap = a*x[i] - x[e] + b*y[i] - y[e]
            if abs(ts[i] - t_chap) < z_chap/2:
                Inliners_count += 1

        if Inliners_count >= 0.5*N**2:
            teta = np.arctan(a/b)
            Un = arr([U_chap, teta]).T
        else:
            print(Inliners_count)
            Un = arr([0, 0])
            teta = 0
        
        #Premirere colonne = length, 2eme = theta
        Un_tab[e] = Un


        # MULTI SPATIAL SCALE MAX POOLING
        # Sigma represente le voisinage spatial. Il nou est donée dans le papier  : 0 to 100 pixels in steps of 10
        if not np.array_equal(Un, [0, 0]):
            bestU, bestTeta = compute_best_flow(x, y, ts,e, Un_tab)
        
        #update vx, vy
        vx_tab[e] = Un_tab[e,0] * np.cos( Un_tab[e,1])
        vy_tab[e] = Un_tab[e,0] * np.sin( Un_tab[e,1])

    return Un_tab, vx_tab, vy_tab

            

def compute_best_flow(x, y, t,e, Un_tab, tpast=500):
    indices = np.where(t <= tpast)[0]

    U_means = []
    tetas_means = []
    for r in range(0, 100, 10):
        # vérifier si l'indices est bon 
        indices = indices[np.where((x[e] - r <= x[indices] & (x[e] + r >= x[indices])))[0]]
        indices= indices[np.where((y[e] - r <= y[indices] & (x[e] + r >= y[indices])))[0]]

        sum_un = 0
        sum_teta = 0
        for i in indices:
            sum_un += Un_tab[i,0]
            sum_teta += Un_tab[i,1]

        U_means.append(sum_un/len(indices))
        tetas_means.append(sum_un/len(indices))
    
    sig_max = np.argmax(U_means)
    Un_tab[e] = np.array([U_means[sig_max], tetas_means[sig_max]])
    return U_means[sig_max], tetas_means[sig_max]










# v = local_plane_fitting(x,y,p,ts,5, 100)
s = compute_local_flow(x, y, p, ts)
# print(len(s[0]))

data = np.load("test.npy")

# C = compute_lmsq(data[:,0], data[:,1], data[:,2], True)
# print(C)

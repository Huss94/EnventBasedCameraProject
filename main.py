import scipy.io
import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import bisect
import os 

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


def local_plane_fitting(x, y, ts, nei, show_plane = False):
    th1 = 1e-5
    th2 = 0.05

    Pi0 = compute_lmsq(x[nei], y[nei], ts[nei], show_plane)


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

        Pi = compute_lmsq(x[nei], y[nei], ts[nei], show_plane)
        if Pi is not None:
            eps = np.linalg.norm(arr(Pi) - arr(Pi0))
            Pi0 = Pi
        else:
            break
    return Pi0


def compute_flow(x, y, p, ts, N=3, dt=1000, N_events = None):
    # Sauvegarde des metrics
    if N_events is None: 
        N_events = len(x)


    local_length_teta = np.zeros((len(x),2))
    # vx_tab_local = np.zeros(len(x))
    # vy_tab_local = np.zeros(len(x))

    corrected_length_teta = np.zeros((len(x),2))
    # vx_tab_corrected = np.zeros(len(x))
    # vy_tab_corrected = np.zeros(len(x))


    #Conversion du temps en seconde
    ts_sec = ts*10**(-6)


    pbar = tqdm(range(N_events))

    # Boucle a travers tous les evements
    for e in pbar:

        # COMPUTE LOCAL FLOW

        # On défini la fenetre temporelle avec une recherche par dichotomie étant donné que le temps est croissant
        # Faire un np.where allonge le temps de calcul et est inutile dans ce cas la 
        # En procédant ainsi on passe de 1h de temps de calcul à environ 10min pour 1255559 points
        up_indice = bisect.bisect_right(ts, ts[e] + dt)
        down_indice = bisect.bisect_left(ts, ts[e] - dt)
        time_window = np.arange(down_indice, up_indice, 1)

        # Recheche des voisins dans la time_window.
        nei = time_window[np.where((x[e] - N <= x[time_window]) & (x[time_window] <= x[e] + N))[0]]
        nei = nei[np.where((y[e] - N <= y[nei]) & (y[nei] <= y[e] + N))[0]]


        # le plan ax + by + t + c = 0
        P = local_plane_fitting(x, y, ts_sec, nei,False)
        if P is None:
            continue
        a, b, c = P


        Inliners_count = 0
        z_chap = np.sqrt(a**2 + b**2)


        # Û in the article
        speed_amplitude = 1 / z_chap 

        for i in nei:
            # the formula is given as below in the paper but I think it is an error because t_chap would be too high every time (except for event with low coordinates)
            # t_chap1  = (a*x[i] - x[e]) + (b*y[i] - y[e])

            # What I think they wanted to write :
            t_chap = a*(int(x[i]) - int(x[e])) + b*(int(y[i]) - int(y[e]))
            if abs((ts_sec[i] - ts_sec[e]) - t_chap) < z_chap/2:
                Inliners_count += 1

        if Inliners_count >= 0.5*N**2:
            teta = np.arctan2(a,b)
            Un = arr([speed_amplitude, teta]).T
        else:
            Un = arr([0, 0])
            teta = 0
        
        #Premirere colonne = length, 2eme = theta
        local_length_teta[e] = Un


        # MULTI SPATIAL SCALE MAX POOLING
        # Sigma represente le voisinage spatial. Il nou est donée dans le papier  : 0 to 100 pixels in steps of 10
        if not np.array_equal(Un, [0, 0]):
            #FLow correction
            bestU, bestTeta, _ = correct_flow(x, y, ts,e, local_length_teta)
            corrected_length_teta[e,:] = arr([bestU, bestTeta])

            # We should probably assign the corrected flow to all the events in the spatial scale, as said in the paper:
            # "Third, we calculate the mean direction for the flows in this scale and assign the direction to all the local flow events within this scale"



            
    # On retourne une concaténation par colonnes de toute les metrics utile :
    # Pour cela on rajoute des dimensions aux vecteur:
    x,y,ts,p = add_axis(x,y,ts,p)


    # On ne sauvegarde pas les vitesse (on peut les retrouver avec l'angle et la longueur)
    returned_array = np.concatenate((x,y,ts,p, local_length_teta, corrected_length_teta), axis = 1)
    return returned_array[0:N_events, :]

def add_axis(*tab):
    return_tab = []
    for i in tab:
        if len(i.shape) == 1: #Sécruité
            i = i[:, None]
            return_tab.append(i)
    return return_tab


def correct_flow(x, y, t,e, Un_tab, tpast=500):

    up_indice = bisect.bisect_right(ts, ts[e] + tpast)
    down_indice = bisect.bisect_left(ts, ts[e] - tpast)
    time_window = np.arange(down_indice, up_indice, 1)


    U_means = []
    tetas_means = []
    scale_indices = []
    for r in range(10, 100, 10):
        indices = time_window[np.where((x[e] - r <= x[time_window]) & (x[time_window] <= x[e] + r))[0]]
        indices = indices[np.where((y[e] - r <= y[indices]) & (y[indices] <= y[e] + r))[0]]
        scale_indices.append(indices)
        

        sum_un = 0
        sum_teta = 0
        for i in indices:
            sum_un += Un_tab[i,0]
            sum_teta += Un_tab[i,1]

        if len(indices) != 0:
            U_means.append(sum_un/len(indices))
            tetas_means.append(sum_teta/len(indices))
        else:
            U_means.append(0)
            tetas_means.append(0)
        
    
    sig_max = np.argmax(U_means)
    return U_means[sig_max], tetas_means[sig_max], scale_indices[sig_max]






# Chargement des données
mat = scipy.io.loadmat('multipattern1.mat')
x = mat['x'][:,0] 
y = mat['y'][:,0] 
p = mat['p'][:,0]
ts = mat['ts'][:,0]


# Calcul le flow local et le flow corrigé, retourne un tableau avec toutes les metriques
print("Calcul du flow des events : ")
metrics = compute_flow(x, y, p, ts, N = 3 , dt =  1000)


if not os.path.isdir("data"):
    os.mkdir("data")

# écriture des données dans un fichier texte : 
print("Ecriture des données dans le fichier out_flow.txt")
with open("data/out_flow.txt", 'w') as f:
    for i in range(metrics.shape[0]):
        s = ''
        space = ' '
        for j in range(metrics.shape[1]):
            if j < 4: 
                val = int(metrics[i,j])
            else: 
                val = round(metrics[i,j], 5)
                if j == metrics.shape[1] - 1:
                    space = '\n'

            s+= str(val) + space

        f.write(s)

# On sauvegarde en np.save aussi (beaucoup plus rapide a load)

np.save("data/out_flow.npy", metrics)
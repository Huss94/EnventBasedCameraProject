import cv2 as cv
import numpy as np 
import scipy.io
import bisect
import time

def visu_event(x,y,ts,p, affiche_time):
    ind_min = 0
    ind_max = 0
    h = max(y) + 1
    w = max(x) + 1
    step = 0
    while (ind_max < len(x) - 10):
        print(f"{ind_min} / {len(x)}")

        im = np.ones((w,h))  * 0.5
        ind_max = bisect.bisect_right(ts,ts[ind_min] + affiche_time)
        indices = np.arange(ind_min, ind_max)

        im[x[indices], y[indices]] = p[indices]
        ind_min = ind_max
        cv.imshow("event", im)
        cv.waitKey(int(affiche_time * 10**(-2)))

def visu_flow(x,y,ts,p, EDL, ARMS, affiche_time):

    ind_min = 0
    ind_max = 0
    h = max(y) + 1
    w = max(x) + 1
    step = 0
    while (ind_max < len(x) - 10):
        print(f"{ind_min} / {len(x)}")


        EDL_im = np.zeros((w,h,3), dtype = np.uint8)
        # EDL_im[:,:, 2] = 150

        ARMS_im= np.zeros((w,h,3), dtype = np.uint8)
        # ARMS_im[:,:, 2] = 150


        ind_max = bisect.bisect_right(ts,ts[ind_min] + affiche_time)
        indices = np.arange(ind_min, ind_max)

        H_EDL = np.uint16(EDL[indices,1] * 255/(2*np.pi))
        H_EDL = H_EDL[:, np.newaxis]
        H_ARMS = np.uint16(ARMS[indices,1] * 255/(2*np.pi))
        H_ARMS= H_ARMS[:, np.newaxis]
        
        fill = 255* np.ones_like(H_EDL)
        fill_edl = fill.copy()
        fill_edl[np.where(H_EDL == 0)] = 0

        fill_arms= fill.copy()
        fill_arms[np.where(H_ARMS == 0)] = 0

        EDL_im[x[indices], y[indices]] = np.concatenate((H_EDL,fill_edl,fill_edl), axis = 1)
        ARMS_im[x[indices], y[indices]] = np.concatenate((H_ARMS,fill_arms,fill_arms), axis = 1)

        ind_min = ind_max

        EDL_im = cv.cvtColor(EDL_im, cv.COLOR_HSV2BGR)
        ARMS_im= cv.cvtColor(ARMS_im, cv.COLOR_HSV2BGR)


        print("EDL theta mean : ", np.mean(H_EDL[np.where(H_EDL != 0)]))
        print("ARMS tehta means : ", np.mean(H_ARMS[np.where(H_ARMS != 0)]))

        cv.imshow("ARMS", ARMS_im)
        cv.imshow("EDL", EDL_im)
        cv.waitKey(0)





# PROCESSING THE DATA
#=========================================
#0, 1, 2, 3,     4,         5,          6,          7
#x, y, t, p, EDL_length, EDL_theta, ARMS_length, ARMS_theta
#=========================================

# s = np.loadtxt("out_flow.txt", dtype = np.float64)
# np.load est bien plus rapide que de parser le fichier txt

# si True, on utilise les données qu'on a généré en utilisant l'algorithime en CPP de l'auteur 
cpp_file = False

if cpp_file:
    events = np.load("data/events_cpp.npy")
else:
    events = np.load("data/out_flow.npy")


x,y,ts,p =  np.uint16(events[:, 0]), np.uint16(events[:, 1]), np.uint64(events[:, 2]), np.int8(events[:, 3])
p[np.where(p == -1)] = 0


# IF LOAD CPP
if cpp_file : 
    ARMS= events[:,4:6]
    ARMS[:,1] = ARMS[:,1]%(2*np.pi)

    EDL= events[:,8:10]
    EDL[:,1] = EDL[:,1]%(2*np.pi)


else:
    EDL = events[:,4:6]
    EDL[:,1] = EDL[:,1]%(2*np.pi)

    ARMS = events[:,6:]
    ARMS[:,1] = ARMS[:,1]%(2*np.pi)


visu_flow(x,y,ts,p,EDL,ARMS,16666)


# Modifyng polarity

print("wait")

# visu(x,y,ts,p, 16666)

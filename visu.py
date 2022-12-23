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


        #HSV gray Image
        im = np.zeros((w,h,3), dtype = np.uint8)
        im[:,:, 2] = 150


        ind_max = bisect.bisect_right(ts,ts[ind_min] + affiche_time)
        indices = np.arange(ind_min, ind_max)

        H = np.uint16(EDL[indices,1] * 255/(2*np.pi))
        H = H[:, np.newaxis]
        fill = 255* np.ones_like(H)

        im[x[indices], y[indices]] = np.concatenate((H,fill,fill), axis = 1)

        ind_min = ind_max

        im = cv.cvtColor(im, cv.COLOR_HSV2BGR)
        cv.imshow("flow", im)
        cv.waitKey(0)





# PROCESSING THE DATA
#=========================================
#0, 1, 2, 3,     4,         5,          6,          7
#x, y, t, p, EDL_length, EDL_theta, ARMS_length, ARMS_theta
#=========================================

# np.load est bien plus rapide que de parser le fichier txt
# s = np.loadtxt("out_flow.txt", dtype = np.float64)
events = np.load("data/out_flow.npy")
# events = np.load("data/events_cpp.npy")

x,y,ts,p =  np.uint16(events[:, 0]), np.uint16(events[:, 1]), np.uint64(events[:, 2]), np.int8(events[:, 3])
p[np.where(p == -1)] = 0


# IF LOAD CPP
# ARMS= events[:,4:6]
# ARMS[:,1] = ARMS[:,1]%(2*np.pi)

# EDL= events[:,8:10]
# EDL[:,1] = EDL[:,1]%(2*np.pi)



EDL = events[:,4:6]
EDL[:,1] = EDL[:,1]%(2*np.pi)

ARMS = events[:,6:]
ARMS[:,1] = ARMS[:,1]%(2*np.pi)


visu_flow(x,y,ts,p,EDL,ARMS,30000)


# Modifyng polarity

print("wait")

# visu(x,y,ts,p, 16666)

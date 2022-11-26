import numpy as np
import scipy.io
from event import event

def load_event(path):
    mat = scipy.io.loadmat(path)
    x = mat['x']
    y = mat['y']
    p = mat['p']
    t = mat['ts']
    ev = []
    for i in range(len(x)):
        ev.append(event(x[i,0],y[i,0],t[i,0],p[i,0]))


    return ev



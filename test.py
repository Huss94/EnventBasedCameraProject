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

#ON convertie le temps en miliseconde
ts = mat['ts']

f = open('events.txt', 'w')

for i in range(len(x)):
    if i % 1000 == 0:
        print(f"{i}/{len(x)}")
    f.write(f"{x[i][0]} {y[i][0]} {ts[i][0]} {p[i][0]}\n")
    
f.close()
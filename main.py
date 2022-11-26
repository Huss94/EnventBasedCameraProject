import scipy.io
import numpy as np
from utils import *
import time

events = np.array(load_event('s.mat'))
for e in events:
    print(np.where(events.x < 10))


# for ind, e in enumerate(events):



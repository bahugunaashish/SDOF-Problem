# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 19:39:24 2024

@author: abahugu
"""

import numpy as np
from ProcessNGA import *
import matplotlib.pyplot as plt
from NewmarkBetaMethod import *
from numpy.linalg import inv
from numpy import linalg as LA
from matplotlib.pyplot import figure
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#%% #%% Problem #1 a and b

filepath1 = ('C:/Users/abahugu/OneDrive - Clemson University/Course-Work/Earhquake engg/Homeworks/HW#3/Gilroy #1 Loma Prieta 1989/PEERNGARecords_Unscaled/RSN765_LOMAP_G01090.AT2')

filepath2 = ('C:/Users/abahugu/OneDrive - Clemson University/Course-Work/Earhquake engg/Homeworks/HW#3/Rinaldi Nothridge 1994/PEERNGARecords_Unscaled/RSN1063_NORTHR_RRS318.AT2')

paths = [filepath1, filepath2]

for p in paths:
    desc, npts, dt, time, inp_acc = processNGAfile(p, scalefactor=None)

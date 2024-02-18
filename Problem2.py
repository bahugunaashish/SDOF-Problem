# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:05:30 2024

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

#%% Problem #2 a and b
print('*********Problem #2 (a)**********')
# Normalization mode shape
print('\n\n=======================================================')
# Homework 1 and 2 condensed stifness matirx
inch = 1.
ksi = 1.
kip = 1.
ft = 12*inch
sec = 1. 
H1 = 15*ft
H2 = 12*ft

L = 20*ft

E  = 29000*ksi
Ib1 = 7800*inch**4
Ib2 = 7800*inch**4
Ic1 = 2660*inch**4
Ic2 = 2140*inch**4

K11 = 2*(12*E*Ic1/(H1)**3) + 2*(12*E*Ic2/(H2)**3)
K21 = 2*((-12*E*Ic2/(H2)**3)) 
K31 = (6*E*Ic1/(H1)**2) - (6*E*Ic2/(H2)**2)
K41 = K31
K51 = -(6*E*Ic2/(H2)**2)
K61 = -(6*E*Ic2/(H2)**2)

K12 = K21
K22 = 2*(12*E*Ic2/(H2)**3)
K32 = (6*E*Ic2/(H2)**2)
K42 = (6*E*Ic2/(H2)**2)
K52 = (6*E*Ic2/(H2)**2)
K62 = (6*E*Ic2/(H2)**2)


K13 = K31
K23 = K32
K33 = (4*E*Ib1/L) + (4*E*Ic1/H1) + (4*E*Ic2/H2)
K43 = (2*E*Ib1/L)
K53 = (2*E*Ic2/H2)
K63 = 0.

K14 = K41
K24 = K42
K34 = K43
K44 = (4*E*Ib1/L) + (4*E*Ic1/H1) + (4*E*Ic2/H2)
K54 = 0.
K64 = (2*E*Ic2/H2)

K15 = K51
K25 = K52
K35 = K53
K45 = K54
K55 = (4*E*Ib2/L) + (4*E*Ic2/H2)
K65 = (2*E*Ib2/L)

K16 = K61
K26 = K62
K36 = K63
K46 = K64
K56 = K65
K66 = (4*E*Ib2/L) + (4*E*Ic2/H2)


K = np.array([[K11, K12, K13, K14, K15, K16], 
             [K21, K22, K23, K24, K25, K26],
             [K31, K32, K33, K34, K35, K36],
             [K41, K42, K43, K44, K45, K46],
             [K51, K52, K53, K54, K55, K56],
             [K61, K62, K63, K64, K65, K66]]
             )
print(f' \nSystem Stiffness Matrix :\n {K}')

Kuu = K[0:2, 0:2]
print(f'\n Kuu: \n{Kuu}')

Kuc = K[0:2, 2:6]
print(f'\n Kuc: \n{Kuc}')

Kcc = K[2:6, 2:6]
print(f'\n Kcc:\n{Kcc}')

Kcu = np.transpose(Kuc)
print(f'\n Kcu: \n{Kcu}')

 
#np.matmul(a, b)
Ke = Kuu - np.matmul(np.matmul(Kuc, inv(Kcc)), Kcu)

print(f'\n Equivalent stiffness matrix \n Ke : \n{Ke}')

W1 = 40*kip
W2 = 20*kip
g = 386.08*inch/sec**2
zeta = 0.02
mass = np.array([[W1/g , 0.], 
                [0.,    W2/g]])

print(f'\nMass Matrix [M]: \n {mass}')



eigenvalues, eigenvectors = LA.eig(np.matmul(inv(mass), Ke))

print(f'Eigen Values: \n {eigenvalues}')
print(f'\n Eigen vectors: \n {eigenvectors}')

wn = np.sqrt(eigenvalues)
print(f'\nFrequency :\n {wn}')

Tn = 2*np.pi/wn
print(f'\nPeriod :\n {Tn}')


phiNorm = np.zeros((2, 2))
for i in range (len(Tn)):
    Denume = np.sqrt(np.matmul(np.matmul(np.transpose(eigenvectors[:,i]), mass), eigenvectors[:,i]))
    phiNorm[:,i] = (eigenvectors[:,i]/Denume)
    
print(f'\nMass Normalised Mode shape: \n {phiNorm}\n')

for i in range (2):
    print(f'**Mode {-(i-1*2)}**')
    for j in range (2):
        print(f'{phiNorm[j][i]:.2f}')
    

mr = np.matmul(np.matmul(np.transpose(phiNorm), mass), phiNorm) 
Kr = np.matmul(np.matmul(np.transpose(phiNorm), Ke), phiNorm) 
print(f'\nMass Matrix Mr: \n{mr}')
print(f'\nMatrix Kr: \n{Kr}')

#mapping vector
DirV = [[1.], 
     [1.]
     ]
# L = [1.,1.]
     
DirV = np.array(DirV)
# DirVr = np.matmul(np.matmul(np.transpose(phiNorm), mass), DirV)
# Gammar = np.matmul(inv(mr), DirVr)
Gammar = np.matmul(np.matmul(np.transpose(phiNorm), mass), DirV)
print(f'\n Mass Participation factor: \n{Gammar}')
# print(f'\n Mass Participation factor: \n {Gammar1}')
print('\n==============================================================\n')


# P#2b
ModeStiff = Kr
ModeDamp = np.diag(2*zeta*wn)
ModeMass = mr
print(f'Modal Mass: \n{mr} \n Mode Damping: \n{ModeDamp} \nstiffness: \n{ModeStiff}')
print(f'\n Mass Participation factor: \n{Gammar}')

print('\n*********Problem #2 (b)**********')

filepath1 = ('RSN765_LOMAP_G01090.AT2')

filepath2 = ('RSN1063_NORTHR_RRS318.AT2')

paths = [filepath1, filepath2]

method = 'Average'

EQ = np.array([0, 1])
# nsteps = 7998
for j in EQ:
    if j==1:
        nsteps = 1991
        nmodes = 2
    else:
        nsteps = 7998
        nmodes = 2
        

    Zu = np.zeros((nmodes, nsteps))
    
    Zv = np.zeros((nmodes, nsteps))
    
    Za = np.zeros((nmodes, nsteps))
    
    
    for i in range (nmodes):
        # dt = 0.01
        m = ModeMass[i][i]
        k = ModeStiff[i][i]
        c = ModeDamp[i][i]
        desc, npts, dt, time, inp_acc = processNGAfile(paths[EQ[j]], scalefactor=None)
        p = Gammar[i]*inp_acc*g
        Zu[i][:], Zv[i][:], Za[i][:], dynStiffness, a, b, t =  NewmarkBetaMethod(m, k, c, p, dt, method, flag=1) 
        t = np.append(t, t[-1]+dt)
        
        print(f'Modal Disp , Velo, Acc for Mode {(i+1)}** for EQ {desc[0:11]}')
        # print(f'Max Displacement = {MaxD:.2f} in \nMax Velocity = {MaxV:.2f} in/sec \nMax Acceleration = {MaxA:.2f} g')
        # Create subplots
        plt.figure(i)
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    
        # Plot on the first subplot
        axs[0].plot(t, Za[i][:], color='blue', linewidth=1., label='Acceleration')
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Za (in/sec2)")
        axs[0].grid(color='grey', linestyle='--', linewidth=0.5)
        axs[0].set_xlim([0., abs(time[-1])])
        axs[0].legend()
    
        # Plot on the second subplot
        axs[1].plot(t, Zv[i][:], color='blue', linewidth=1., label='Velocity')
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Zv (in)")
        axs[1].grid(color='grey', linestyle='--', linewidth=0.5)
        axs[1].set_xlim([0., abs(time[-1])])
        axs[1].legend()
    
        # Plot on the third subplot
        axs[2].plot(t, Zu[i][:], color='blue', linewidth=1., label='Displacement')
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Zu (in)")
        axs[2].grid(color='grey', linestyle='--', linewidth=0.5)
        axs[2].set_xlim([0., abs(time[-1])])
        axs[2].legend()
    
        # Display the plot
        plt.show()
        
    U = np.matmul(phiNorm, Zu)
    V = np.matmul(phiNorm, Zv)
    A = np.matmul(phiNorm, Za)
    
    # print(U)
    for i in range (2): 
        MaxD = max(abs(U[i][:]))
        MaxV = max(abs(V[i][:]))
        MaxA = max(abs(A[i][:]))/g
        print(f'Max values of Disp , Velo, Acc for DOF {(i+1)}** for EQ {desc[0:11]}')
        print(f'Max Displacement = {MaxD:.2f} in \nMax Velocity = {MaxV:.2f} in/sec \nMax Acceleration = {MaxA:.2f} g')
        # Create subplots
        plt.figure(i)
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    
        # Plot on the first subplot
        axs[0].plot(t, A[i][:], color='blue', linewidth=1., label='Acceleration')
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Acce. (in/sec2)")
        axs[0].grid(color='grey', linestyle='--', linewidth=0.5)
        axs[0].set_xlim([0., abs(time[-1])])
        axs[0].legend()
    
        # Plot on the second subplot
        axs[1].plot(t, V[i][:], color='blue', linewidth=1., label='Velocity')
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Vel. (in)")
        axs[1].grid(color='grey', linestyle='--', linewidth=0.5)
        axs[1].set_xlim([0., abs(time[-1])])
        axs[1].legend()
    
        # Plot on the third subplot
        axs[2].plot(t, U[i][:], color='blue', linewidth=1., label='Displacement')
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Disp. (in)")
        axs[2].grid(color='grey', linestyle='--', linewidth=0.5)
        axs[2].set_xlim([0., abs(time[-1])])
        axs[2].legend()
    
        # Display the plot
        plt.show()
        
    rotationDisp = np.matmul(np.matmul(-inv(Kcc),Kcu), U)
    for i in range (4):
        MaxRD = max(abs(rotationDisp[i][:]))
        print(f'Max rotation of Dof {i+3} for EQ {desc[0:11]}: \n{MaxRD} (rad)')
        print(f'Actual rotation for dof {(i+3)}** for EQ {desc[0:11]}')
        # print(f'Max Displacement = {MaxD:.2f} in \nMax Velocity = {MaxV:.2f} in/sec \nMax Acceleration = {MaxA:.2f} g')
        # Create subplots
        plt.figure(i)
        # fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
        # Adjust layout
        plt.tight_layout()
        # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    
        # Plot on the first subplot
        plt.plot(t, rotationDisp[i][:], color='blue', linewidth=1., label='Rotation dof'+str(i+3))
        plt.xlabel("Time (s)")
        plt.ylabel("Rotation (rad)")
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.xlim([0., abs(time[-1])])
        plt.legend()
               
        # Display the plot
        plt.show()

# print(Zu)



    
    
    
    
    
    

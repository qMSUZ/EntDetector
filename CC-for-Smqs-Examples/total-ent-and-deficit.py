#! /usr/bin/python
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2021 -- 2022 by Marek Sawerwain                         *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                              by Joanna Wiśniewska                       *
# *                                  <Joanna.Wisniewska@wat.edu.pl>         *
# *                                                                         *
# *                              by Roman Gielerak                          *
# *                                  <R.Gielerak@issi.uz.zgora.pl>          *
# *                                                                         *
# *   Part of the EntDetector:                                              *
# *         https://github.com/qMSUZ/EntDetector                            *
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 3 of the License, or     *
# *   (at your option) any later version.                                   *
# *                                                                         *
# *   This program is distributed in the hope that it will be useful,       *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
# *   GNU General Public License for more details.                          *
# *                                                                         *
# *   You should have received a copy of the GNU General Public License     *
# *   along with this program; if not, write to the                         *
# *   Free Software Foundation, Inc.,                                       *
# *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
# ***************************************************************************/



import os
import numpy as np
import scipy.special

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cmath 

#import contextlib

import multiprocessing
import time
#import more_itertools


from entdetector import *

def total_entropy_and_deficit(d, nqubits, ntasks):
    #d = 2
    #nqubits = 2
    N = nqubits
    
   
    #q0 = create_qubit_zero_state()
    #q12 = create_qubit_bell_state()
    #q3 = create_qubit_zero_state()
    #qreg = np.kron(q0, np.kron(q12, q3))
    
    qreg1= create_ghz_state( d, nqubits )
    qreg2= create_wstate( nqubits )
    
    entropysup = 0.0
    for k in range(1, int(np.floor(N/2))+1, 1 ):
        entropysup = entropysup + ( scipy.special.binom(np.floor(N/2), k) * np.log(d ** k) )
    
    
    p=0.0;
    entDeficitTbl=[]
    totalentropyTbl=[]
    totalp=[]
    while p<1.0:
        qreg = p * qreg1 + (1-p)*qreg2
        qreg = qreg / np.linalg.norm(qreg)
    
        #print_state(qreg)
    
        #schmidt_shape=(4, 4)
        #s, e, f = schmidt_decomposition_for_vector_pure_state(qreg, schmidt_shape)
        #print("Number of schmidt coeff =", len(s))
        #print("Schmidt coeff =",chop(s)); print()
    
        #qden = vector_state_to_density_matrix( qreg )
        #eval = entropy(qden)
        #print("entropy of qden = ",eval)
    
        #detection_entanglement_by_paritition_division(qreg, nqubits)
    
        total_entropy_val=0
        total_negativity_val = 0.0
    
        [r,idxtoremove,total_entropy_val,total_negativity_val] = entropy_by_paritition_division_parallel_v0( qreg, nqubits, ntasks, 0 )
        #[r,idxtoremove,total_entropy_val,total_negativity_val] = entropy_by_paritition_division_serial( qreg, nqubits, 0 )
    
        #print("Total Entropy = ", total_entropy_val, " sup ", entropysup)
        #print("Total Negativity = ", total_negativity_val)
        totalp.append( p )
        totalentropyTbl.append( total_entropy_val )
        entDeficitTbl.append( d+(entropysup - total_entropy_val) )
        p=p+0.0125
    
    return totalp, totalentropyTbl, entDeficitTbl


def calculate_total_entropy_and_deficit(d, nqubits, ntasks):
    
    # totalp_N=0
    # totalentropyTbl_N=0
    # entDeficitTbl_N=0
    
    start = time.perf_counter()

    # p=0.5


    # qreg1= create_ghz_state( d, nqubits )
    # qreg2= create_wstate( nqubits )

    # qreg = p * qreg1 + (1-p)*qreg2
    # qreg = qreg / np.linalg.norm(qreg)


    # r,idxtoremove,total_entropy_val,total_negativity_val = entropy_by_paritition_division_parallel_v0( qreg, nqubits, ntasks, 0 )
    
    totalp_N, totalentropyTbl_N, entDeficitTbl_N = total_entropy_and_deficit(d, nqubits, ntasks) 

    
    finish = time.perf_counter()
    
    print()
    print(f'Finished in {round(finish-start,2)} seconds(s): routine: total_entropy_and_deficit({d}, {nqubits}, ntasks={ntasks})')
    
    return totalp_N, totalentropyTbl_N, entDeficitTbl_N 


def create_figure():
    ntasks=4

    totalp_N2, totalentropyTbl_N2, entDeficitTbl_N2 =  calculate_total_entropy_and_deficit(2, 2, ntasks)
    totalp_N3, totalentropyTbl_N3, entDeficitTbl_N3 = calculate_total_entropy_and_deficit(2, 3, ntasks)
    totalp_N4, totalentropyTbl_N4, entDeficitTbl_N4 = calculate_total_entropy_and_deficit(2, 4, ntasks)
    totalp_N5, totalentropyTbl_N5, entDeficitTbl_N5 = calculate_total_entropy_and_deficit(2, 5, ntasks)

    print(); print(); print();

    
    plt.cla()
    plt.clf()
    plt.plot(totalp_N2, totalentropyTbl_N2, 'r:', label='Total Entropy N=2')
    plt.plot(totalp_N2, entDeficitTbl_N2, 'g:', label='Deficit of entanglement N=2')
    plt.plot(totalp_N3, totalentropyTbl_N3, 'r-.', label='Total Entropy N=3')
    plt.plot(totalp_N3, entDeficitTbl_N3, 'g-.', label='Deficit of entanglement N=3')
    plt.plot(totalp_N4, totalentropyTbl_N4, 'r--', label='Total Entropy N=4')
    plt.plot(totalp_N4, entDeficitTbl_N4, 'g--', label='Deficit of entanglement N=4')
    plt.plot(totalp_N5, totalentropyTbl_N5, 'r.-', label='Total Entropy N=5')
    plt.plot(totalp_N5, entDeficitTbl_N5, 'g.-', label='Deficit of entanglement N=5')
    #plt.plot(totalp_N10, totalentropyTbl_N10, 'r.-', label='Total Entropy N=10')
    #plt.plot(totalp_N10, entDeficitTbl_N10, 'g.-', label='Deficit of entanglement N=10')
    #plt.plot(totalp_NT, totalentropyTbl_NT, 'r.-', label='Total Entropy N=T')
    #plt.plot(totalp_NT, entDeficitTbl_NT, 'g.-', label='Deficit of entanglement N=T')
    plt.xlabel("Value of parameter t")
    plt.ylabel("Value of Total Entropy and Deficit")
    plt.legend(ncol=2, bbox_to_anchor=(1.075, 1.4))
    #plt.legend(bbox_to_anchor=(1.0, 1.0))
    ##
    plt.savefig('total-ent-and-deficit.eps', bbox_inches='tight')
    #    


if __name__ == '__main__':
    create_figure()

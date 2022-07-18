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
# * file is related to the paper titled                                     *
# *   Classical Computer assisted  analysis  of small multi-qudit systems   *
# *    published in IEEE Access Vol.???, No.???, pages ?????, Year          *
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

def entropy_by_paritition_division_for_given_partition_list(queue, q, lp, nqubits, verbose ):
    #s = q.size
    s = nqubits
    # generation of all two partitite divisions of given
    # set which is build from the quantum register q
    total_entropy_val = 0.0
    total_negativity_val = 0.0
    entropy_val = 0.0
    res = [ ]
    idxtoremove = [ ]
#    k = [0] * s
#    M = [0] * s
#    p = 2
#    partititon_p_initialize_first(k, M, p)
#    lp = []
#    lp = lp + [make_partititon_as_list(k)]
#    while partition_p_next(k, M, p):
#        lp = lp + [make_partititon_as_list(k)]
    for i in lp:
            if verbose==1 or verbose==2:
                    print(i[0], i[1])
            mxv=2**len(i[0])
            myv=2**len(i[1])
            if verbose==1:
                    print(mxv,"x",myv)
            #m=qcs.Matrix(mxv, myv)
            m  = np.zeros((mxv, myv), dtype=complex)
            #mt=qcs.Matrix(mxv, myv)
            mt = np.zeros((mxv, myv), dtype=complex)
            for x in range(0,mxv):
                    for y in range(0, myv):
                            xstr=bin(x)[2:]
                            ystr=bin(y)[2:]
                            xstr='0'*(len(i[0])-len(xstr)) + xstr
                            ystr='0'*(len(i[1])-len(ystr)) + ystr
                            cstr=[0]*s
                            for xidx in range(0, len(xstr)):
                                    idx = i[0][xidx]
                                    cstr[idx]=xstr[xidx]
                            for yidx in range(0, len(ystr)):
                                    idx = i[1][yidx]
                                    cstr[idx]=ystr[yidx]
                            cidx=""
                            for c in cstr:
                                    cidx=cidx+c
                            dcidx=bin2dec(cidx)
                            dxidx=bin2dec(xstr)
                            dyidx=bin2dec(ystr)
                            if verbose==1:
                                    print("D("+xstr+","+ystr+")","D(",dxidx,dyidx,") C",dcidx,cidx,cstr)
                            #m.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            m[dxidx,dyidx] = q[dcidx]
                            #mt.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            mt[dxidx,dyidx] = q[dcidx]
            if verbose==1:
                    #m.PrMatlab()
                    print("m matrix")
                    print(m)
            #mf=m.Calc_D_dot_DT() # D * D'
            mf = m @ m.transpose()
            #sd=mf.SpectralDecomposition()
            #sd.eigenvalues.Chop()
            #ev_count=sd.eigenvalues.NonZeros()
            (ev,evec)=eigen_decomposition(mf)
            ev=chop(ev)
            ev_count = np.count_nonzero(ev)
            if (ev_count > 1):
                entropy_val=0.0
                negativity_val=0.0
                for ee in ev:
                    #e=np.sqrt(ee)
                    e=ee
                    if e != 0.0:
                        entropy_val = entropy_val + (e ** 2) * np.log((e ** 2))
                        #entropy_val = entropy_val + e * np.log2(e)
                        #entropy_val = entropy_val + e * np.log10(e)
                        negativity_val=negativity_val+e
                total_negativity_val += negativity_val
                total_entropy_val += -entropy_val
            
            if verbose==1 or verbose==2:
                    print("non zero:", ev_count)
                    print("ev=",ev)
            if (ev_count==1) and (len(i[0])==1):
                idxtoremove=idxtoremove+i[0]
                
            if (ev_count==1) and (len(i[1])==1):
                idxtoremove=idxtoremove+i[1]
                
            res=res + [[ev_count, [i[0], i[1]]]]
            if verbose==1 or verbose==2:
                print()
    queue.put( (res, idxtoremove, total_entropy_val, 0.5*((total_negativity_val ** 2)-1.0) ) )

def entropy_by_paritition_division_parallel_v0( q, nqubits, ntasks, verbose ):
    #s = q.size
    s = nqubits
    # generation of all two partitite divisions of given
    # set which is build from the quantum register q
    ret_total_entropy_val = 0.0
    ret_total_negativity_val = 0.0
    entropy_val = 0.0
    ret_res = [ ]
    ret_idxtoremove = [ ]
    k = [0] * s
    M = [0] * s
    p = 2

    partititon_p_initialize_first(k, M, p)

    lp = []
    lp = lp + [make_partititon_as_list(k)]
    while partition_p_next(k, M, p):
        lp = lp + [make_partititon_as_list(k)]

    length_of_single_part = len(lp) / float(ntasks)
    list_of_parts = [ ]
    num_of_elements=0

    while num_of_elements < len(lp):
        list_of_parts.append(lp[int(num_of_elements):int(num_of_elements + length_of_single_part)])
        num_of_elements += length_of_single_part

    queue = multiprocessing.Queue()
    tasks = range(ntasks)
    
    for part in list_of_parts:
        p=multiprocessing.Process(target=entropy_by_paritition_division_for_given_partition_list, args=(queue, q, part, nqubits, verbose))
        p.start()
        
    for _ in tasks:
        (r,idxtoremove,total_entropy_val,total_negativity_val) = queue.get()
        ret_res = ret_res + r
        ret_idxtoremove = ret_idxtoremove + idxtoremove
        ret_total_entropy_val = ret_total_entropy_val + total_entropy_val
        ret_total_negativity_val = ret_total_negativity_val + total_negativity_val

    return ret_res, ret_idxtoremove, ret_total_entropy_val, 0.5*((ret_total_negativity_val ** 2)-1.0)


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

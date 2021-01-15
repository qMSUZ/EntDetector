#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2020 -- 2021 by Marek Sawerwain                         *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                              by Joanna Wiśniewska                       *
# *                                  <Joanna.Wisniewska@wat.edu.pl>         *
# *                                                                         *
# *                              by Marek Wróblewski                        *
# *                                  <M.Wroblewski@issi.uz.zgora.pl>        *
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


"""
First version created on Sat Nov 21 18:33:49 2020

@author: Marek Sawerwain
"""

import numpy as np
import scipy
import cvxopt

import random as rd

import math


class DimensionError(Exception):
    """DimensionError"""
    pass

# code based on chop
# discussed at:
#   https://stackoverflow.com/questions/43751591/does-python-have-a-similar-function-of-chop-in-mathematica
def chop(expr, delta=10 ** -10):
    if isinstance(expr, (int, float, complex)):
        return 0 if -delta <= expr <= delta else expr
    else:
        return [chop(x) for x in expr]

#
# basic state creation
#
def create_base_state(d, n, base_state):
    v = np.zeros(d ** n)
    v[base_state] = 1
    return v

def create_pure_state(d, n, base_state):
    return create_base_state(d, n, base_state)

def create_qubit_zero_state():
    v = np.zeros(2)
    v[0] = 1.0
    return v

def create_qubit_one_state():
    v = np.zeros(2)
    v[1] = 1.0
    return v

def create_qubit_plus_state():
    v = np.zeros(2)
    v[0] = 1.0 / np.sqrt(2)
    v[1] = 1.0 / np.sqrt(2)
    return v

def create_qubit_minus_state():
    v = np.zeros(2)
    v[0] =   1.0 / np.sqrt(2)
    v[1] = - 1.0 / np.sqrt(2)
    return v

def create_qutrit_state(base_state):
    v = np.zeros(3)
    v[base_state] = 1.0
    return v

def create_qutrit_zero_state():
    v = np.zeros(2)
    v[0] = 1.0
    return v

def create_qutrit_one_state():
    v = np.zeros(2)
    v[0] = 1.0
    return v

def create_qutrit_two_state():
    v = np.zeros(2)
    v[0] = 1.0
    return v

def create_qutrit_plus_state():
    v = np.ones(3)
    v[0] = 1.0
    return v

def create_qubit_bell_state(minus=0):
    d = 2
    n = 2
    v = np.zeros(d ** n)
    v[0] = 1.0 / np.sqrt(2)
    if minus == 1:
        v[(d ** n) - 1] = -1.0 / np.sqrt(2)
    else:
        v[(d ** n) - 1] = 1.0 / np.sqrt(2)
    return v

def create_mixed_state(d,n):
    qden = np.eye(d ** n) / (d ** n)
    return qden

"""
state |00..0> +  |kkk...k>
where k = d - 1 and d is demension of single qudit of quantum register
with n equally dimensional qudits
"""
def create_0k_stat(d, n):
    v = np.zeros(d ** n)
    v[0] = 1.0/np.sqrt(2)
    v[-1] = v[0]
    return v

def create_max_entangled_pure_state(d):
    v = np.reshape( np.eye(d), d**2 )
    v = v / np.sqrt( d )
    return v

def create_bes_horodecki_24_state(b):
    x = np.array([b, b, b, b, b, b, b, b])
    rho = np.diag(x, k=0)
    rho[4][4] = (1.0 + b) / 2.0
    rho[7][7] = (1.0 + b) / 2.0
    rho[4][7] = np.sqrt(1.0 - b * b) / 2.0
    rho[7][4] = np.sqrt(1.0 - b * b) / 2.0
    rho[5][0] = b
    rho[6][1] = b
    rho[7][2] = b
    rho[0][5] = b
    rho[1][6] = b
    rho[2][7] = b
    rho = rho / (7.0 * b + 1.0)
    return rho

def create_bes_horodecki_33_state(a):
    x = np.array([a, a, a, a, a, a, a, a, a])
    rho = np.diag(x, k=0)
    rho[6][6] = (1.0 + a) / 2.0
    rho[8][8] = (1.0 + a) / 2.0
    rho[8][6] = np.sqrt(1.0 - a * a) / 2.0
    rho[6][8] = np.sqrt(1.0 - a * a) / 2.0
    rho[4][0] = a
    rho[8][0] = a
    rho[4][8] = a
    rho[0][4] = a
    rho[0][8] = a
    rho[8][4] = a
    rho = rho / (8.0 * a + 1.0)
    return rho

def create_ghz_state(d, n):
    g = np.zeros(d ** n)
    step = np.sum(np.power(d, range(n)))
    g[range(d) * step] = 1/np.sqrt(d)
    return g

def create_wstate(n):
    w = np.zeros(2 ** n)
    for i in range (n):
        w[2 ** i] = 1 / np.sqrt(n)
    return w

def create_isotropic_qubit_state(p):
    q = create_qubit_bell_state()
    qdentmp = np.outer(q, q)
    qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    return qden

def create_werner_two_qubit_state(p, state="Bell+"):
    if state=="Bell+":
        q = create_qubit_bell_state()
    if state=="Bell-":
        q = create_qubit_bell_state(minus=1)
    if state=="W":
        q = create_wstate(2)
    qdentmp = np.outer(q, q)
    qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    return qden

def create_x_two_qubit_random_state():
    antydiagval = np.random.rand(2)

    diagval = np.random.rand(4)
    diagval = (diagval / np.linalg.norm(diagval)) ** 2

    leftVal0=diagval[1] * diagval[2]
    rightVal0=np.abs(antydiagval[1]) ** 2

    leftVal1=diagval[0] * diagval[3]
    rightVal1=np.abs(antydiagval[0]) ** 2

    while not (leftVal0 >= rightVal0 and leftVal1 >= rightVal1):
        antydiagval = np.random.rand(2)

        diagval = np.random.rand(4)
        diagval = (diagval / np.linalg.norm(diagval)) ** 2

        leftVal0=diagval[1] * diagval[2]
        rightVal0=np.abs(antydiagval[1]) ** 2

        leftVal1=diagval[0] * diagval[3]
        rightVal1=np.abs(antydiagval[0]) ** 2

    qden = np.zeros( 16 )
    qden = np.reshape( qden, (4, 4) )

    qden[0,0] = diagval[0]
    qden[1,1] = diagval[1]
    qden[2,2] = diagval[2]
    qden[3,3] = diagval[3]

    qden[0,3] = antydiagval[0]
    qden[1,2] = antydiagval[1]
    qden[2,1] = antydiagval[1].conj()
    qden[3,0] = antydiagval[0].conj()

    return qden




#
#
#

def vector_state_to_density_matrix(q):
    return np.outer(q, q)

def create_density_matrix_from_vector_state(q):
    return vector_state_to_density_matrix(q)

#
# Spectral decomposition of density matrix
#

def eigen_decomposition(qden):
    eigval, eigvec = np.linalg.eigh(qden)
    return eigval, eigvec

def eigen_decomposition_for_pure_state(q):
    qden = np.outer(q,q)
    eigval,eigvec = np.linalg.eigh(qden)
    return eigval, eigvec

def reconstruct_density_matrix_from_eigen_decomposition(eigval, eigvec):
    i = 0
    qden = np.zeros([eigval.shape[0],eigval.shape[0]])
    for ev in eigval:
        qden = qden + np.outer(eigvec[:, i], ev * eigvec[:, i])
        i = i + 1
    return qden

#
# Schmidt decomposition of vector state
#

def schmidt_decomposition_for_vector_pure_state(q, decomposition_shape):
    d1,d2 = decomposition_shape
    m = q.reshape(d1, d2)
    u, s, vh = np.linalg.svd(m, full_matrices=True)
    
    return s, u, vh

def schmidt_decomposition_for_square_operator(qden, decomposition_shape):
    pass

def schmidt_rank_for_vector_pure_state(q, decomposition_shape):
    d1,d2 = decomposition_shape
    m = q.reshape(d1, d2)
    sch_rank = np.linalg.matrix_rank(m)
    return sch_rank

def reconstruct_state_after_schmidt_decomposition(s, e, f):
    dfin = s.shape[0] * e.shape[0]
    v = np.zeros(dfin)

    idx = 0
    for sv in s:
        v = v + np.kron(sv * e[idx], f[idx])
        idx = idx + 1
    return v

#
# Creation of spectral table of given quantum state
# expressed as density matrix
#

def create_spectral_and_schmidt_table(qden, schmidt_shape):
    ev,evec = eigen_decomposition(qden)
    #idxs = [i for i, e in enumerate(ev) if e != 0.0]
    idxs = range(len(ev))
    evtbl=[]
    for ii in idxs:
        evtbl.append( (ev[ii], evec[:, ii]) )
    schmdtbl=[]
    for evt in evtbl:
        s, e, f = schmidt_decomposition_for_vector_pure_state(evt[1], schmidt_shape)
        schmdtbl.append( (s,e,f) )
    return evtbl, schmdtbl

def create_spectral_and_schmidt_table_data(qden, schmidt_shape):
    evtbl, schmdtbl = create_spectral_and_schmidt_table( qden, schmidt_shape)
    return (evtbl, schmdtbl)

def create_sas_table_data(qden, schmidt_shape):
    evtbl, schmdtbl = create_spectral_and_schmidt_table( qden, schmidt_shape)
    return (evtbl, schmdtbl)

def calculate_statistic_for_sas_table(e,s):
    idx=len(e)-1;
    vtbl0=[]
    vtbl1=[]
    while idx >=0:
        vtbl0.append(s[idx][0][0])
        vtbl1.append(s[idx][0][1])
        idx=idx-1
    return ( np.var(vtbl0), np.var(vtbl1), np.std(vtbl0), np.std(vtbl1) )

def print_sas_table( sas_table, statistics=0):
    e,s = sas_table
    idx=len(e)-1;
    vtbl0=[]
    vtbl1=[]
    while idx >=0:
        vtbl0.append(s[idx][0][0])
        vtbl1.append(s[idx][0][1])
        print(chop(s[idx][0]), "|", chop(e[idx][0]))
        idx=idx-1
    if statistics==1:
        print("var=", np.var(vtbl0), np.var(vtbl1))
        print("std=", np.std(vtbl0), np.std(vtbl1))

#
# Routines for Entropy calculation
#

def entropy(qden, logbase="e"):
    if np.isscalar(qden):
        raise DimensionError("Wrong dimension of argument!")
        return None
    eigval,evec = eigen_decomposition(qden)
    entropy_val = 0.0
    for e in eigval:
        if chop(e) != 0:
            if logbase == "e":
                entropy_val = entropy_val + e * np.log(e)
            if logbase == "2":
                entropy_val = entropy_val + e * np.log2(e)
            if logbase == "10":
                entropy_val = entropy_val + e * np.log10(e)
    return chop(-entropy_val)

#
# Negativity
#

def negativity( qden, d=2, n=2 ):
    dim = int(np.log(d ** n)/np.log(d))
    qdentmp = partial_tranpose(qden, [[dim,dim], [dim,dim]], [0, 1])
    negativity_value = (np.linalg.norm(qdentmp, 'nuc') - 1.0)/2.0
    return negativity_value

#
# Concurrence
#

def concurrence( qden ):
    pauliy=np.array([0.0, -1.0J, 1.0J, 0.0]).reshape(2,2)
    qden=np.matrix(qden)
    R = qden * np.kron(pauliy, pauliy) * qden.getH() * np.kron(pauliy, pauliy)
    e,v=np.linalg.eig(R)
    evalRealList = [float(ev.real) for ev in e]
    
    evallist = []
    for v in evalRealList:
        if v>0:
            evallist.append(np.sqrt(v))
        else:
            evallist.append(chop(v))
    evallist=-np.sort(-np.array(evallist))
    c=np.max([evallist[0]-evallist[1]-evallist[2]-evallist[3], 0.0])
    
    return c

#
#
#

# reference implementation directly based on 
# https://github.com/qutip/qutip/blob/master/qutip/partial_transpose.py
# 
def partial_tranpose_main_routine(rho, dims, mask):
    mask = [int(i) for i in mask]
    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate( [ [pt_dims[n, mask[n]] for n in range(nsys)],
                               [pt_dims[n, 1 - mask[n]] for n in range(nsys)] ] )
    data = rho.reshape(np.array(dims).flatten()).transpose(pt_idx).reshape(rho.shape)

    return data

def partial_tranpose(rho, dims, no_tranpose):
    return partial_tranpose_main_routine(rho, dims, no_tranpose)

def partial_tranpose_for_qubit(rho, no_tranpose):
    pass

def partial_tranpose_for_qutrits(rho, no_tranpose):
    pass

#
#
#

#
# directly based on the code and discussion:
#   https://github.com/cvxgrp/cvxpy/issues/563
#
def partial_trace_main_routine(rho, dims, axis=0):
    dims_tmp = np.array(dims)
    reshaped_rho = rho.reshape(np.concatenate((dims_tmp, dims_tmp), axis=None))

    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims) + axis - 1, -1)

    return_trc_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    dims_untraced = np.delete(dims_tmp, axis)
    rho_dim = np.prod(dims_untraced)
    
    return return_trc_out_rho.reshape([rho_dim, rho_dim])

def partial_trace(rho, ntrace_out):
    dimensions = []
    single_dim = int(np.log2(rho.shape[0]))
    for _ in range(int(single_dim)):
        dimensions.append(single_dim)

    densitytraceout = partial_trace_main_routine(rho, dimensions, axis = ntrace_out)
    return densitytraceout


def swap_subsystems_for_bipartite_system(rho, dims):
    finaldim=rho.shape
    rshp_dims=(dims, dims)
    axesno=len([e for l in rshp_dims for e in l])
    axesswap=list(range(axesno))[::-1]
    
    rrho = rho.reshape(np.concatenate(rshp_dims, axis=None))
    rrho = np.transpose(rrho, axes=axesswap )
    orho=rrho.reshape( finaldim ).T
    
    return orho

def permutation_of_subsystems(rho, dims, perm):
    nsys=len(dims)
    finaldim=rho.shape
    rshp_dims=(dims, dims)
    axesno=len([e for l in rshp_dims for e in l])
    axesswap=list(range(axesno))[::-1]
    permaxesswap = np.zeros(axesno)
    parts=int(axesno/nsys)
    idx=0
    while idx<parts:
        bidx=(idx*nsys)
        bendidx=bidx+parts+1
        for ii in range(nsys):
            permaxesswap[bidx:bendidx][ii] = axesswap[bidx:bendidx][perm[-ii]]
        idx=idx+1
    rrho = rho.reshape(np.concatenate(rshp_dims, axis=None))
    rrho = np.transpose(rrho, axes=permaxesswap.astype(int) )
    orho=rrho.reshape( finaldim ).T
    return orho


#
# Gram matrices
#

def gram_right_of_two_qubit_state(v):
    m = np.zeros((2,2))
    m[0,0] = np.abs(v[0])**2 + np.abs(v[1])**2;                m[0,1] = v[0].conjugate()*v[2] + v[1].conjugate()*v[3];
    m[1,0] = v[2].conjugate()*v[0] + v[3].conjugate()*v[1];    m[1,1] = np.abs(v[2])**2 + np.abs(v[3])**2;
    
    return m

def gram_left_of_two_qubit_state(v):
    m = np.zeros((2,2))
    m[0,0] = np.abs(v[0])**2.0 + np.abs(v[2])**2.0;            m[0,1] = v[0].conjugate()*v[1] + v[2].conjugate()*v[3];
    m[1,0] = v[1].conjugate()*v[0] + v[3].conjugate()*v[2];    m[1,1] = np.abs(v[1])**2.0 + np.abs(v[3])**2.0;
    
    return m

def full_gram_of_two_qubit_state(v):
    A = np.abs(v[0])**2.0 + np.abs(v[1])**2.0
    B = np.abs(v[2])**2.0 + np.abs(v[3])**2.0
    C = np.abs(v[0])**2.0 + np.abs(v[2])**2.0
    D = np.abs(v[1])**2.0 + np.abs(v[3])**2.0
    C13 = v[0].conjugate()*v[2] + v[1].conjugate()*v[3]
    C12 = v[0].conjugate()*v[1] + v[2].conjugate()*v[3]
    C31 = v[2].conjugate()*v[0] + v[3].conjugate()*v[1]
    C21 = v[1].conjugate()*v[0] + v[3].conjugate()*v[2]

    m = np.zeros((4,4))

    m[0,0] = A * C;     m[0,1] = A * C12;   m[0,2] = C * C13;   m[0,3] = C13 * C12;
    m[1,0] = A * C21;   m[1,1] = A * D;     m[1,2] = C13 * C21; m[1,3] = D * C13;
    m[2,0] = C31 * C;   m[2,1] = C31 * C12; m[2,2] = B * C;     m[2,3] = B * C12;
    m[3,0] = C31 * C21; m[3,1] = D * C31;   m[3,2] = B * C21;   m[3,3] = B * D;

    return m


def gram_matrices_of_vector_state(v, d1, d2):
    dl = np.zeros((d1,d2))
    for i in range(d1):
        ii=0;
        for j in range(d2):
            idx=(i)*d2+j
            dl[ii,i]= dl[ii,i] + v[idx]
            ii=ii+1
    
    dr = np.zeros((d2,d1))
    for j in range(d2):
        ii=0;
        for i in range(d1):
            idx=(i)*d2+j
            dr[ii,j]= dr[ii,j] + v[idx]
            ii=ii+1
    
    dRprime = np.zeros((d2,d1))
    for i in range(0, d1):
        for j in range(0, d2):
            dRprime[i,j] = dr[i] @ dr[j]
    
    dLprime = np.zeros((d1,d2))
    for i in range(0,d1):
        for j in range(0,d2):
            dLprime[i,j] = dl[i] @ dl[j]
    
    return dRprime, dLprime, np.kron(dRprime, dLprime)

#
#
#

def monotone_for_two_qubit_system(rho):
    # S(1) + S(2) − S(12)
    qr1=partial_trace(rho, 1)
    qr2=partial_trace(rho, 0)
    monotone12 = entropy(qr1) + entropy(qr2) - entropy(rho)
    return monotone12


def monotone_for_three_qubit_system(rho):
    pass

def monotone_for_four_qubit_system(rho):
    pass

def monotone_for_five_qubit_system(rho):
    pass

#
#
#

def create_random_qudit_state(d, n, o=0): # o=0 - only real numbers, 1 - complex numbers, 2 - mixed numbers (~ 1/2 complex numbers)
    ampNumber = d ** n
    psi = np.ndarray(shape=(ampNumber),dtype=complex)
    F = np.ndarray(shape=(ampNumber),dtype=complex)
    if o == 0:
        for i in range(ampNumber):
            F[i] = complex(rd.uniform(-1,1),0)
    elif o == 1:
        for i in range(ampNumber):
            a = rd.uniform(-1,1)
            b = rd.uniform(-1,1)
            F[i] = complex(a,b)
    elif o == 2:
        for i in range(ampNumber):
            a = rd.uniform(-1,1)
            x = rd.randint(0,1)
            if x == 0:
                b = 0
            else:
                b = rd.uniform(-1,1)
            F[i] = complex(a,b)
    else:
        raise ValueError('Option has to be: 0, 1 or 2')
        return None
    #normalization
    con = np.matrix.conjugate(F)
    norm = np.inner(con,F)
    norm = np.sqrt(norm)
    for i in range(ampNumber):
        psi[i] = F[i] / norm
    return psi

#o=0 - only real numbers, 1 - complex numbers, 2 - mixed numbers (~ 1/2 complex numbers)
def create_random_density_state(d, n, o=0):
    if o not in (0,1,2):
        raise ValueError('Option has to be: 0, 1 or 2')
        return None
    else:
        vs = create_random_qudit_state(d,n,o)
        rho = np.outer(vs,np.matrix.conjugate(vs))
        return rho

#o=0 - only real numbers, 1 - complex numbers, 2 - mixed numbers (~ 1/2 complex numbers)
def create_random_density_state_mix(d, n, o=0):
    ampNumber = d ** n
    F = np.ndarray(shape=(ampNumber,ampNumber),dtype=complex)
    if o == 0:
        for i in range(ampNumber):
            for j in range(ampNumber):
                F[j,i] = complex(rd.uniform(-1,1),0)
    elif o == 1:
        for i in range(ampNumber):
            for j in range(ampNumber):
                F[j,i] = complex(rd.uniform(-1,1),rd.uniform(-1,1))
    elif o == 2:
        for i in range(ampNumber):
            for j in range(ampNumber):
                a = rd.uniform(-1,1)
                x = rd.randint(0,1)
                if x == 0:
                    b = 0
                else:
                    b = rd.uniform(-1,1)
                F[j,i] = complex(a,b)
    else:
        print('Option has to be: 0, 1 or 2')
        return 0
    rho = np.add(F, np.matrix.conjugate(F))
    rho = np.divide(rho, 2)
    return rho

#o=0 - only real numbers, 1 - complex numbers, 2 - mixed numbers (~ 1/2 complex numbers)
def create_random_unitary_matrix(dim, o):
    F=np.zeros((dim,dim),dtype=complex)
    #Q=np.zeros((dim,dim),dtype=complex)
    if o==0:
        for i in range(dim):
            for j in range(dim):
                F[j,i]=complex(rd.uniform(0,1)/np.sqrt(2),0)
    elif o==1:
        for i in range(dim):
            for j in range(dim):
                F[j,i]=complex(rd.uniform(0,1),rd.uniform(0,1))/np.sqrt(2)
    elif o==2:
        for i in range(dim):
            for j in range(dim):
                a=rd.uniform(0,1)
                x=rd.randint(0,1)
                if x==0:
                    b=0
                else:
                    b=rd.uniform(0,1)
                F[j,i]=complex(a,b)/np.sqrt(2)
    else:
        print('Option has to be: 0, 1 or 2')
        return 0
    Q,R=np.linalg.qr(F)
    d=np.diagonal(R)
    ph=d/np.absolute(d)
    U=np.multiply(Q,ph,Q)
    return U


#
# small routine for better
# matrix display
#

def pretty_matrix_print(x, _pprecision=4):
    with np.printoptions(precision = _pprecision, suppress=True):
        print(x)

#
# partitions generators
#

def partititon_initialize_first(kappa,M):
    for i in range(0, len(kappa)):
        kappa[i]=0
        M[i]=0

def partititon_initialize_last(kappa,M):
    for i in range(0, len(kappa)):
        kappa[i]=i
        M[i]=i

def partititon_p_initialize_first(kappa, M, p):
    n=len(kappa)
    for i in range(0, n-p+1):
        kappa[i]=0
        M[i]=0
    for i in range(n-p+1, n, 1):
        kappa[i]=i-(n-p)
        M[i]=i-(n-p)

def partititon_p_initialize_last(kappa, M, p):
    n=len(kappa)
    for i in range(0, p):
        kappa[i]=i
        M[i]=i
    for i in range(p, n, 1):
        kappa[i]=p-1
        M[i]=p-1

def partition_size(M):
    n=len(M)
    return M[n-1]-M[0]+1

def partititon_disp(kappa):
        n=len(kappa)
        m=max(kappa)
        fstr=""
        for j in range(0, m+1):
                string='('
                for i in range(0,n):
                        if kappa[i]==j:
                                string=string+str(i)+','
                string=string[0:len(string)-1]
                string=string+')'
                fstr=fstr +string
        return '{'+fstr+'}'


def partititon_as_list(kappa):
        n=len(kappa)
        m=max(kappa)
        fstr=[]
        for j in range(0, m+1):
                string=[]
                for i in range(0,n):
                        if kappa[i]==j:
                                string=string+[i]
                fstr=fstr + [string]
        return fstr


def partition_next(kappa, M):
    n=len(kappa)
    for i in range(n-1, 0, -1):
        if kappa[i] <= M[i-1]:
            kappa[i]=kappa[i]+1
            M[i]=max(M[i], kappa[i])
            for j in range(i+1, n, 1):
                kappa[j]=kappa[0]
                M[j]=M[i]
            return True
    return False

def partition_p_next(kappa, M, p):
        n=len(kappa)
        p=partition_size(M)
        for i in range(n-1,0,-1):
                if kappa[i]<p-1 and kappa[i]<=M[i-1]:
                        kappa[i]=kappa[i]+1
                        M[i]=max(M[i], kappa[i])
                        for j in range(i+1, n-(p-M[i])+1):
                                kappa[j]=0
                                M[j]=M[i]
                        for j in range(n-(p-M[i])+1, n):
                                kappa[j]=p-(n-j)
                                M[j]=p-(n-j)
                        return True
        return False

def gen_all_k_elem_subset(k,n):
    A=[0]*(k+1)
    for i in range(1,k+1):
        A[i]=i
    if k >= n:
        return A[1:]
    output=[]
    p=k
    while p>=1:
        output = output +[A[1:k+1]]
        if A[k]==n:
            p=p-1
        else:
            p=k
        if p>=1:
            for i in range(k,p-1,-1):
                A[i]=A[p]+i-p+1
    return output

# integer l must be a power of 2 
def gen_two_partion_of_given_number(l):
    d = l / 2
    part = []
    r=l/2
    while r!=1:
            part=part + [[int(l/r), int(r)]]
            r=r/2
    return part


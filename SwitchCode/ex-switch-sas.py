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

from entdetector import *
import numpy as np
from scipy.linalg import expm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cmath 

import contextlib

import matplotlib as mpl
mpl.use('PS') 


# global variables
# for amplitudes for quantum states

a0 = 0.0
b0 = 0.0
a1 = 0.0
b1 = 0.0

timetbl = [ ]
entTbl = [ ]
negTbl = [ ]
negTblToCompare = [ ]
maxEigen = [ ]
sumOthEigen = [ ]
conTbl = [ ]
diffTbl = [ ]

elTbl = [ ]
erTbl = [ ]

elTblToCompare = [ ]

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

font2 = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }


def reset_state():
    global a0, b0, a1, b1
    a0 = 0.0
    b0 = 0.0
    a1 = 0.0
    b1 = 0.0

def print_state( state ):
    i=0
    print( "d bbb Amplitude")
    print( "  ABC  ")
    for a in state:
        if a != 0:
            print( i, "{0:03b}".format(i), a)
        i=i+1

def create_uqs_operator(t):
    Uqs_t=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, (1.0+np.cos(np.pi*t)+np.sin(np.pi*t)*1.j)/2.0, 0.0, (1.0-np.cos(np.pi*t)-np.sin(np.pi*t)*1.j)/2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, (1.0-np.cos(np.pi*t)-np.sin(np.pi*t)*1.j)/2.0, 0.0, (1.0+np.cos(np.pi*t)+np.sin(np.pi*t)*1.j)/2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return Uqs_t

def create_udmgs_operator(t, Ds):
    Udmqs_t=np.array([[np.cos(2.0*Ds), 0.0, 0.0, 0.0, 0.0, 0.0, -1.j*np.sin(2.0*Ds), 0.0],
                    [0.0, np.cos(2.0*Ds), 0.0, 0.0, 0.0, 0.0, 0.0, -1.j*np.sin(2.0*Ds)],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, (1.0+np.cos(np.pi*t)+np.sin(np.pi*t)*1.j)/2.0, 0.0, (1.0-np.cos(np.pi*t)-np.sin(np.pi*t)*1.j)/2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, (1.0-np.cos(np.pi*t)-np.sin(np.pi*t)*1.j)/2.0, 0.0, (1.0+np.cos(np.pi*t)+np.sin(np.pi*t)*1.j)/2.0, 0.0, 0.0],
                    [-1.j*np.sin(2.0*Ds), 0.0, 0.0, 0.0, 0.0, 0.0, np.cos(2.0*Ds), 0.0],
                    [0.0, -1.j*np.sin(2.0*Ds), 0.0, 0.0, 0.0, 0.0, 0.0, np.cos(2.0*Ds)]])
    return Udmqs_t

def promote_to_real_amplitudes(q):
    for r,c in np.ndenumerate(q):
        q[r] = np.linalg.norm(q[r])
    return q


def density_matrix_promote_to_real_values(qden):
    qreal = np.zeros( qden.shape )
    for x in range(qden.shape[0]):
        for y in range(qden.shape[1]):
            qreal[x,y] = np.linalg.norm( qden[x,y] )
    return qreal


qA=create_qubit_plus_state()
#qB=create_qubit_plus_state()
#qA=create_base_state(2, 1, 0)
qB=create_base_state(2, 1, 0)

q0=create_base_state(2, 1, 0)
q1=create_base_state(2, 1, 1)

def calculate_negativity_for_ABC_pairs(qden):
    qABden = partial_trace_main_routine(qden, [2, 2, 2], axis=2)
    qACden = partial_trace_main_routine(qden, [2, 2, 2], axis=1)
    qBCden = partial_trace_main_routine(qden, [2, 2, 2], axis=0)
    negvalqAB = negativity(qABden)
    negvalqAC = negativity(qACden)
    negvalqBC = negativity(qBCden)
    return negvalqAB, negvalqAC, negvalqBC

def calculate_concurrence_for_ABC_pairs(qden):
    qABden = partial_trace_main_routine(qden, [2, 2, 2], axis=2)
    qACden = partial_trace_main_routine(qden, [2, 2, 2], axis=1)
    qBCden = partial_trace_main_routine(qden, [2, 2, 2], axis=0)
    negvalqAB = concurrence(qABden)
    negvalqAC = concurrence(qACden)
    negvalqBC = concurrence(qBCden)
    return negvalqAB, negvalqAC, negvalqBC

def calculate_entropy_for_ABC(qden):
    qAden = partial_trace_main_routine(partial_trace_main_routine(qden, [2, 2, 2], axis=2), [2, 2], axis=1)
    qBden = partial_trace_main_routine(partial_trace_main_routine(qden, [2, 2, 2], axis=0), [2, 2], axis=1)
    qCden = partial_trace_main_routine(partial_trace_main_routine(qden, [2, 2, 2], axis=0), [2, 2], axis=0)
    eval = entropy(qden)
    evalqA = entropy(qAden)
    evalqB = entropy(qBden)
    evalqC = entropy(qCden)
    return eval, evalqA, evalqB, evalqC

def calculate_entropy_separability_for_ABC(qden, logbase="e"):
    entleft=0
    entright=0
    schmidt_shape=(2, 4)
    e,s = create_sas_table_data(qden, schmidt_shape)
    idx=len(e)-1;
    while idx >=0:
        eval = e[idx][0]
        if eval!=0.0:
            if chop(eval) >= precision_for_entrpy_calc:
                if logbase == "e":
                    entright = entright + eval * np.log(eval)
                if logbase == "2":
                    entright = entright + eval * np.log2(eval)
                if logbase == "10":
                    entright = entright + eval * np.log10(eval)
            if s[idx][0][0]!=0.0 and s[idx][0][1]!=0.0:
                entleft = entleft + eval * ( (s[idx][0][0])**2 * (np.log(s[idx][0][0]**2)) + (s[idx][0][1])**2 * (np.log(s[idx][0][1]**2)) )
            if s[idx][0][0]==0.0 and s[idx][0][1]!=0.0:
                entleft = entleft + eval * ( (s[idx][0][1])**2 * (np.log(s[idx][0][1]**2)) )
            if s[idx][0][0]!=0.0 and s[idx][0][1]==0.0:
                entleft = entleft + eval * ( (s[idx][0][0])**2 * (np.log(s[idx][0][0]**2)) )
        idx=idx-1

    return -entleft, -entright

def sas_for_entry_state():
    uqsopr = create_uqs_operator(0.0)

    qreg = np.kron(np.kron(qA, qB), q1)

    qregaftersw = np.matmul(uqsopr, qreg)
    qregaftersw=promote_to_real_amplitudes(qregaftersw)
    qregaftersw = chop(qregaftersw)

    qden = vector_state_to_density_matrix( qregaftersw )
    nvAB, nvAC, nvBC = calculate_negativity_for_ABC_pairs(qden)
    print("Negativity for AB=", nvAB, "AC=", nvAC, " BC=", nvBC)
    eval, evA, evB, evC = calculate_entropy_for_ABC(qden)
    print("entropy for all: ", eval, "A=", evA, "B=", evB, " C=", evC)

    el,er=calculate_entropy_separability_for_ABC(qden)
    print("entropy separability: ", el, er, "sep? ",el <= er)

    schmidt_shape=(2, 4)
    e,s = create_sas_table_data(qden, schmidt_shape)
    print("SAS for switch for entry state")
    print_sas_table( (e,s) )
    print("initial state")
    print_state(qreg)
    print("state after uqs for t=0")
    print_state(qregaftersw)

def sas_for_final_state():
    uqsopr = create_uqs_operator(1.0)

    qreg = np.kron(np.kron(qA, qB), q1)

    qregaftersw = np.matmul(uqsopr, qreg)
    qregaftersw=promote_to_real_amplitudes(qregaftersw)
    qregaftersw = chop(qregaftersw)

    qden = vector_state_to_density_matrix( qregaftersw )
    nvAB, nvAC, nvBC = calculate_negativity_for_ABC_pairs(qden)
    print("Negativity for AB=", nvAB, "AC=", nvAC, " BC=", nvBC)
    eval, evA, evB, evC = calculate_entropy_for_ABC(qden)
    print("entropy for all: ", eval, "A=", evA, "B=", evB, " C=", evC)

    el,er=calculate_entropy_separability_for_ABC(qden)
    print("entropy separability: ", el, er, "sep? ",el <= er)

    schmidt_shape=(2, 4)
    e,s = create_sas_table_data(qden, schmidt_shape)
    print("SAS for switch for entry state")
    print_sas_table( (e,s) )
    print("initial state")
    print_state(qreg)
    print("state after uqs for t=1")
    print_state(qregaftersw)


def sas_for_state_at_t(t):
    print("switch for t=", t)
    qreg = np.kron( np.kron(qA, qB), q1 )
    uqsopr_for_t = create_uqs_operator(t)
    qreg_for_t = np.matmul(uqsopr_for_t, qreg)

    qreg_for_t=promote_to_real_amplitudes(qreg_for_t)

    qden = vector_state_to_density_matrix( qreg_for_t )

    nvAB, nvAC, nvBC = calculate_negativity_for_ABC_pairs(qden)
    print("Negativity for AB=", nvAB, "AC=", nvAC, " BC=", nvBC)

    nvAB, nvAC, nvBC = calculate_concurrence_for_ABC_pairs(qden)
    print("Concurrence for AB=", nvAB, "AC=", nvAC, " BC=", nvBC)

    eval, evA, evB, evC = calculate_entropy_for_ABC(qden)
    print("entropy for all: ", eval, ", A=", evA, ", B=", evB, ", C=", evC)

    el,er=calculate_entropy_separability_for_ABC(qden)
    print("entropy separability: ", el, er, "sep? ",el <= er)

    schmidt_shape = (2, 4)
    e, s = create_sas_table_data(qden, schmidt_shape)
    print( "SAS for switch state at t" )
    print_sas_table( (e, s) )
    print()
    print("initial state")
    print_state(qreg)
    print("state after uqs for t", t)
    print_state(qreg_for_t)
    print("-"*64)


def prepare_data_figure():
    t=0
    tstep=0.1
    
    qreg = np.kron( np.kron(qA, qB), q1 )
    while t<=1.0:
        timetbl.append(t)
        #uqsopr_for_t = create_uqs_operator(t)
        uqsopr_for_t = create_udmgs_operator(t, 0.0)
        qreg_for_t = np.matmul(uqsopr_for_t, qreg)
        qreg_for_t=promote_to_real_amplitudes(qreg_for_t)
        qden = vector_state_to_density_matrix( qreg_for_t )

        nvAB, nvAC, nvBC = calculate_negativity_for_ABC_pairs(qden)
        negTbl.append( nvAB )
        el,er=calculate_entropy_separability_for_ABC(qden)
        elTbl.append( el - er)
        #erTbl.append( er )

        t = t + tstep


def negativity_for_p_t(p, t):
    qreg = np.kron( np.kron(qA, qB), q1 )

    uqsopr_for_t = create_uqs_operator(t)
    qreg_for_t = np.matmul(uqsopr_for_t, qreg)
    qreg_for_t=promote_to_real_amplitudes(qreg_for_t)
    qden = vector_state_to_density_matrix( qreg_for_t )

    qdennew = (p * qden) + ((1-p) * ((1.0/8.0) * np.eye(8)))
    nvAB, nvAC, nvBC = calculate_negativity_for_ABC_pairs(qdennew)

    return nvAB

def negativity_for_t_Ds(t, Ds):
    qreg = np.kron( np.kron(qA, qB), q1 )

    uqsopr_for_t = create_udmgs_operator(t, Ds)
    qreg_for_t = np.matmul(uqsopr_for_t, qreg)
    qreg_for_t=promote_to_real_amplitudes(qreg_for_t)
    qden = vector_state_to_density_matrix( qreg_for_t )

    nvAB, nvAC, nvBC = calculate_negativity_for_ABC_pairs(qden)

    return nvAB

def prop60_for_p_t(p, t):
    qreg = np.kron( np.kron(qA, qB), q1 )

    uqsopr_for_t = create_uqs_operator(t)
    qreg_for_t = np.matmul(uqsopr_for_t, qreg)
    qreg_for_t=promote_to_real_amplitudes(qreg_for_t)
    qden = vector_state_to_density_matrix( qreg_for_t )

    qdennew = (p * qden) + ((1-p) * ((1.0/8.0) * np.eye(8)))
    el,er=calculate_entropy_separability_for_ABC(qdennew)

    return  el - er

def prop60_for_t_Ds(t, Ds):
    qreg = np.kron( np.kron(qA, qB), q1 )

    uqsopr_for_t = create_udmgs_operator(t, Ds)
    qreg_for_t = np.matmul(uqsopr_for_t, qreg)
    qreg_for_t=promote_to_real_amplitudes(qreg_for_t)
    qden = vector_state_to_density_matrix( qreg_for_t )

    el,er=calculate_entropy_separability_for_ABC(qden)

    return  el - er

def negativity_prepare_data_and_create_figure_for_p_and_t():
    step=0.0125

    t = np.arange(0.0, 1.0, step)
    p = np.arange(0.0, 1.0, step)

    X, Y = np.meshgrid( t, p )
    Z = np.zeros( (len(t), len(p)) )
    for x in range(len(t)):
        for y in range(len(p)):
            Z[x][y]=negativity_for_p_t( t[x], p[y] )

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow, linewidth=1, antialiased=True)

    ax.set_xlabel(r'Value of t')
    ax.set_ylabel(r'Value of p')
    ax.set_zlabel(r'Negativity')
    plt.title(r'Value of Negativity')
    #plt.show()

    fig.savefig( 'fig-value-of-negativity.eps' )

def negativity_prepare_data_and_create_figure_for_t_and_Ds():
    step=0.05

    t = np.arange(0.0, 1.0, step)
    Ds = np.arange(0.0, 1.0, step)

    X, Y = np.meshgrid( t, Ds )
    Z = np.zeros( (len(t), len(Ds)) )
    for x in range(len(t)):
        for y in range(len(Ds)):
            Z[x][y] = negativity_for_t_Ds( t[x], Ds[y] )

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow, linewidth=1, antialiased=True)

    ax.set_xlabel(r'Value of t')
    ax.set_ylabel(r'Value of Ds')
    ax.set_zlabel(r'Negativity')
    plt.title(r'Value of Negativity with DM')
    #plt.show()

    fig.savefig( 'fig-value-of-negativity-with-ds.eps' )

def prop60_prepare_data_and_create_figure_for_p_and_t():
    step=0.0125

    t = np.arange(0.0, 1.0, step)
    p = np.arange(0.0, 1.0, step)

    X, Y = np.meshgrid( t, p )
    Z = np.zeros( (len(t), len(p)) )
    for x in range(len(t)):
        for y in range(len(p)):
            Z[x][y]=prop60_for_p_t( t[x], p[y] )

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow, linewidth=1, antialiased=True)

    ax.set_xlabel(r'Value of t')
    ax.set_ylabel(r'Value of p')
    ax.set_zlabel(r'Entropy Difference')
    plt.title(r'Value of Entropy Difference')
    #plt.show()

    fig.savefig( 'fig-value-of-prop-60.eps' )

def prop60_prepare_data_and_create_figure_for_t_and_Ds():
    step=0.05

    t = np.arange(0.0, 1.0, step)
    Ds = np.arange(0.0, 1.0, step)

    X, Y = np.meshgrid( t, Ds )
    Z = np.zeros( (len(t), len(Ds)) )
    for x in range(len(t)):
        for y in range(len(Ds)):
            Z[x][y] = prop60_for_t_Ds( t[x], Ds[y] )

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow, linewidth=1, antialiased=True)

    ax.set_xlabel(r'Value of t')
    ax.set_ylabel(r'Value of Ds')
    ax.set_zlabel(r'Entropy Difference')
    plt.title(r'Value of Entropy Difference with DM')
    #plt.show()

    fig.savefig( 'fig-value-of-prop-60-with-DM.eps' )

def create_figure_switch_and_sas():
    # create figure
    plt.cla()
    plt.clf()

    #plt.plot(ptbl, entTbl, 'r-', label='Entropy')
    #plt.plot(timetbl, negTbl, 'g-s', markevery=1, label='Negativity')
    plt.plot(timetbl, negTbl, 'g-', label='Negativity')
    plt.plot(timetbl, elTbl, 'b--', label = 'EL(t)')
    #plt.plot(timetbl, erTbl, 'b:', label = 'ER')
    #plt.plot(ptbl, conTbl, 'r:*', markevery=5, label='Concurrence')
    #plt.plot(ptbl, diffTbl, 'r:o',markevery=13, label='Difference')
    ##plt.plot(ptbl, diffTbl, linestyle=(0, (3, 5, 1, 5)), label='Difference')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    #plt.axvline(0.33, color='gray',linestyle='--')
    ##plt.axhline(0.025, color='gray',linestyle='--')
    #plt.axvline(0.00, color='red',linestyle=':')
    #plt.text(0.175, -1.20,"Value of $p$",fontdict=font)
    #plt.text(-0.25, -0.90,"State is separable",fontdict=font2)
    #plt.text(-0.2,  -1.05,"$-0.33 \leq p \leq 0.33$",fontdict=font2)
    #plt.text(0.36,  -0.90,"State is entangled",fontdict=font2)
    #plt.text(0.45,  -1.05,"$p > 0.33$",fontdict=font2)
    plt.savefig('entropy-sep-new-crit.eps', bbox_inches='tight')
    plt.show()


def milburn_equation_for_decoherence():
    t=0.0
    tstep=0.1
    Ds=0.25
    gamma = 0.5

    ev=np.array([-1, 0, 0, 0,-2*Ds, -2*Ds, 2*Ds, 2*Ds])

    evec=np.zeros((8,8))
    oldvec=np.array(
        [   [ 0,  0, 0, -1, 0, 1, 0, 0],
            [ 0,  0, 0,  1, 0, 1, 0, 0],
            [ 0,  0, 0,  0, 1, 0, 0, 0],
            [ 0,  0, 1,  0, 0, 0, 0, 0],
            [ 0, -1, 0,  0, 0, 0, 0, 1],
            [-1,  0, 0,  0, 0, 0, 1, 0],
            [ 0,  1, 0,  0, 0, 0, 0, 1],
            [ 1,  0, 0,  0, 0, 0, 1, 0]
        ])

    for idx in range(8):
        evec[idx] = oldvec[idx]/np.linalg.norm(oldvec[idx])
    Hqs = evec.T @ np.diag(ev) @ evec
    #pretty_matrix_print(Hqs)

    qreg = np.kron( np.kron(qA, qB), q1 )
    qden0 = vector_state_to_density_matrix( qreg )

    while t<=5.0:
        timetbl.append(t)
        rho=np.zeros((8,8))
        for m in range(8):
            for n in range(8):
                rho = rho + ( np.exp((-(gamma*t*np.pi)/2 * (ev[m] - ev[n])**2 - 1j*(ev[m]-ev[n])*t*np.pi)) * (evec[m] @ (qden0 @ evec[n]) * np.outer(evec[m], evec[n])) )
    
        #print("trace(rho)=", np.trace(rho))
        #pretty_matrix_print(rho)


        # expm(-1j * np.pi * t * Hqs)
        uqsopr_for_t = create_udmgs_operator(t, Ds)
        qreg_for_t = np.matmul(uqsopr_for_t, qreg)
        qreg_for_t=promote_to_real_amplitudes(qreg_for_t)
        qdenToCompare = vector_state_to_density_matrix( qreg_for_t )
        nvAB2, nvAC2, nvBC2 = calculate_negativity_for_ABC_pairs(qdenToCompare)
        negTblToCompare.append( nvAB2 )
        el,er=calculate_entropy_separability_for_ABC(qdenToCompare)
        elTblToCompare.append( el - er)
        #pretty_matrix_print(qdenToCompare)

        #qrslt = qdenToCompare - rho
        #print(chop(qrslt))
        #nvAB1, nvAC1, nvBC1 = calculate_negativity_for_ABC_pairs(qdenToCompare)
        qden = density_matrix_promote_to_real_values(rho)
        nvAB2, nvAC2, nvBC2 = calculate_negativity_for_ABC_pairs(qden)
        negTbl.append( nvAB2 )
        el,er=calculate_entropy_separability_for_ABC(qden)
        elTbl.append( el - er)
        #print(nvAB1, nvAC1, nvBC1)
        #print(nvAB2, nvAC2, nvBC2)
        t=t+tstep

        plt.cla()
        plt.clf()

        #plt.plot(ptbl, entTbl, 'r-', label='Entropy')
        #plt.plot(timetbl, negTbl, 'g-s', markevery=1, label='Negativity')
        plt.plot(timetbl, negTblToCompare, 'g:', label='Negativity O')
        plt.plot(timetbl, elTblToCompare, 'b:', label = 'EL(t) O')
        plt.plot(timetbl, negTbl, 'g-', label='Negativity')
        plt.plot(timetbl, elTbl, 'b--', label = 'EL(t)')
        #plt.plot(timetbl, erTbl, 'b:', label = 'ER')
        #plt.plot(ptbl, conTbl, 'r:*', markevery=5, label='Concurrence')
        #plt.plot(ptbl, diffTbl, 'r:o',markevery=13, label='Difference')
        ##plt.plot(ptbl, diffTbl, linestyle=(0, (3, 5, 1, 5)), label='Difference')
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        #plt.axvline(0.33, color='gray',linestyle='--')
        ##plt.axhline(0.025, color='gray',linestyle='--')
        #plt.axvline(0.00, color='red',linestyle=':')
        #plt.text(0.175, -1.20,"Value of $p$",fontdict=font)
        #plt.text(-0.25, -0.90,"State is separable",fontdict=font2)
        #plt.text(-0.2,  -1.05,"$-0.33 \leq p \leq 0.33$",fontdict=font2)
        #plt.text(0.36,  -0.90,"State is entangled",fontdict=font2)
        #plt.text(0.45,  -1.05,"$p > 0.33$",fontdict=font2)
        plt.savefig('milburn-for-long-t.eps', bbox_inches='tight')
        #plt.show()




#sas_for_entry_state()
#sas_for_final_state()

#sas_for_state_at_t(0.2)
#sas_for_state_at_t(0.3)
#sas_for_state_at_t(0.4)

#sas_for_state_at_t(0.1)
#sas_for_state_at_t(0.5)
#sas_for_state_at_t(0.9)

#prepare_data_figure()
#create_figure_switch_and_sas()


#negativity_prepare_data_and_create_figure_for_p_and_t()
#prop60_prepare_data_and_create_figure_for_p_and_t()

#negativity_prepare_data_and_create_figure_for_t_and_Ds()
#prop60_prepare_data_and_create_figure_for_t_and_Ds()

milburn_equation_for_decoherence()

#qreg = np.kron(np.kron(qA, qB), q1)
#uqsopr_for_t = create_uqs_operator(0.5)
#qreg_for_t = np.matmul(uqsopr_for_t, qreg)

#qden = vector_state_to_density_matrix( qreg_for_t )
#print(qreg_for_t)
#pretty_matrix_print(qden)


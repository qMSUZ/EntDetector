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
# *   SwitchCode is a part of the EntDetector:                              *
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

import random as r

import contextlib

import matplotlib as mpl
mpl.use('PS')

def count_negativity(t_max, t_gr, a0, b0, a1, b1):
    X=np.array([])
    Y=np.array([])
    Yalt=np.array([])

    t=0.0
    AB1=np.array([0.0, a0*a1, 0.0, a0*b1, 0.0, a1*b0, 0.0, b0*b1])

    while(t <= t_max+t_gr ):
        Uqs_t=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, (1.0+np.cos(np.pi*t)+np.sin(np.pi*t)*1.j)/2.0, 0.0, (1.0-np.cos(np.pi*t)-np.sin(np.pi*t)*1.j)/2.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, (1.0-np.cos(np.pi*t)-np.sin(np.pi*t)*1.j)/2.0, 0.0, (1.0+np.cos(np.pi*t)+np.sin(np.pi*t)*1.j)/2.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        tol = 1e-15
        Uqs_t.real[abs(Uqs_t.real) < tol] = 0.0
        Uqs_t.imag[abs(Uqs_t.imag) < tol] = 0.0
        state=np.matmul(Uqs_t,AB1)
        state_con=np.conjugate(state)
        rho=np.outer(state,state_con)
        Tr3=np.array([[ rho[0][0]+rho[1][1], rho[0][2]+rho[1][3], rho[0][4]+rho[1][5], rho[0][6]+rho[1][7] ],
                      [ rho[2][0]+rho[3][1], rho[2][2]+rho[3][3], rho[2][4]+rho[3][5], rho[2][6]+rho[3][7] ],
                      [ rho[4][0]+rho[5][1], rho[4][2]+rho[5][3], rho[4][4]+rho[5][5], rho[4][6]+rho[5][7] ],
                      [ rho[6][0]+rho[7][1], rho[6][2]+rho[7][3], rho[6][4]+rho[7][5], rho[6][6]+rho[7][7] ]])

        Tr3.real[abs(Tr3.real) < tol] = 0.0
        Tr3.imag[abs(Tr3.imag) < tol] = 0.0

        rho_TA=np.array([[Tr3[0][0], Tr3[0][1], Tr3[2][0], Tr3[2][1]],
                         [Tr3[1][0], Tr3[1][1], Tr3[3][0], Tr3[3][1]],
                         [Tr3[0][2], Tr3[0][3], Tr3[2][2], Tr3[2][3]],
                         [Tr3[1][2], Tr3[1][3], Tr3[3][2], Tr3[3][3]]])
        w,v=np.linalg.eig(rho_TA)
        w.real[abs(w.real) < tol] = 0.0
        N_rho=0.0
        i=0
        while (i<4):
            pom=w[i]
            pom2=pom.real
            if pom2<0.0:
                N_rho+=pom2
            i+=1
        N_rho=abs(N_rho)
        if N_rho<tol:
            N_rho=0.0
        Neg=np.sqrt((pow(np.sin(np.pi*t),2))*(pow(abs(a1*b0 - a0*b1),4))/4.0)
        if Neg<tol:
            Neg=0.0
        X    = np.append(X, t)
        
        Y    = np.append(Y, N_rho)
        Yalt = np.append(Yalt, Neg)
        
        t+=t_gr
    return X,Y,Yalt

def test_for_one_basic_state():
    aa0=1.0/np.sqrt(2.0)
    bb0=1.0/np.sqrt(2.0)
    aa1=1.0
    bb1=0.0
    step=1.0/50.0
    X1,Y1,Z1=count_negativity(1.0, step, aa0,bb0,aa1,bb1)
    plt.plot(X1, Y1, '--')
    plt.xlabel("Time")
    plt.ylabel("Value of negativity measure")
    plt.legend()
    plt.savefig('negativity-basic.eps')
    plt.show()


def test_for_128_states():
    for i in range(128):
        x1=r.uniform(0,1)
        x2=1.0-x1
        aa0=np.sqrt(x1)
        bb0=np.sqrt(x2)
        x1=r.uniform(0,1)
        x2=1.0-x1
        aa1=np.sqrt(x1)
        bb1=np.sqrt(x2)
        step=r.uniform(0,0.25)/10.0
        X1,Y1,Z1=count_negativity(1.0, step, aa0,bb0,aa1,bb1)
        plt.plot(X1, Y1, '.')
    plt.xlabel("Time")
    plt.ylabel("Negativity")
    plt.legend()
    plt.savefig('negativity-128.eps')
    plt.show()

test_for_one_basic_state()
test_for_128_states()

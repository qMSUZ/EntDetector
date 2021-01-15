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


import matplotlib.pyplot as plt
import numpy as np
import contextlib

from entdetector import *

import matplotlib as mpl
mpl.use('PS') 

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


typeOfState="Bell+"
#typeOfState="Bell-"
#typeOfState="W"

with open("output-"+typeOfState+"-sasd.txt", "w") as h, contextlib.redirect_stdout(h):
    print("Type of state", typeOfState)
    print()
    d=2
    p=-0.33
    #p=0.0
    step=0.01;
    ptbl=[]
    entTbl=[]
    negTbl=[]
    conTbl=[]
    diffTbl=[]
    maxEigen=[]
    sumOthEigen=[]
    schmidt_shape=(2, 2)
    while p<=1.0:
        #q = create_qubit_bell_state()
        #qdentmp = np.outer(q, q)
        #qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
        qden = create_werner_two_qubit_state(p, typeOfState)
        e,s = create_sas_table_data(qden, schmidt_shape)
        print("p=%.2f" % p)
        pretty_matrix_print(qden)
        print_sas_table(e,s)
        #(v0,v1,s0,s1) = calculate_statistic_for_sas_table(e,s)
        #print("p=%.2f" % p, "v=%.4f" % v0, "%.4f" % v1, "s=%.4f" % s0, "%.4f" % s1)
        entTbl.append(entropy(qden))
        print("entropy=", entropy(qden))
        negTbl.append(negativity(qden))
        print("negativity=", negativity(qden))
        conTbl.append(concurrence(qden))
        print("concurrence=", concurrence(qden))
        ptbl.append(p)
        maxEigen.append( max([row[0] for row in e]) )
        sumOthEigen.append( sum([row[0] for row in e]) - max([row[0] for row in e]) ) 
        print("max eigen=", maxEigen[-1], "sum of other eigs=", sumOthEigen[-1])
        print("difference between max eigen and sum oth evsl=", maxEigen[-1] - sumOthEigen[-1])
        diffTbl.append(maxEigen[-1] - sumOthEigen[-1])
        p=p+step
        print()
        print()

    p=1.0
    #q = create_qubit_bell_state()
    #qdentmp = np.outer(q, q)
    #qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    schmidt_shape=(2, 2)
    qden = create_werner_two_qubit_state(p, typeOfState)
    e,s = create_sas_table_data(qden, schmidt_shape)
    print("p=%.2f" % p)
    pretty_matrix_print(qden)
    print_sas_table(e,s)
    #(v0,v1,s0,s1) = calculate_statistic_for_sas_table(e,s)
    #print("p=%.2f" % p, "v=%.4f" % v0, "%.4f" % v1, "s=%.4f" % s0, "%.4f" % s1)
    entTbl.append(entropy(qden))
    print("entropy=", entropy(qden))
    negTbl.append(negativity(qden))
    print("negativity=", negativity(qden))
    conTbl.append(concurrence(qden))
    print("concurrence=", concurrence(qden))
    ptbl.append(p)
    maxEigen.append( max([row[0] for row in e]) )
    sumOthEigen.append( sum([row[0] for row in e]) - max([row[0] for row in e]) ) 
    print("max eigen=", maxEigen[-1], "sum of other eigs=", sumOthEigen[-1])
    print("difference between max eigen and sum oth evsl=", maxEigen[-1] - sumOthEigen[-1])
    diffTbl.append(maxEigen[-1] - sumOthEigen[-1])
    print()
    print()



# create figure


plt.cla()
plt.clf()

plt.plot(ptbl, entTbl, 'r-', label='Entropy')
plt.plot(ptbl, negTbl, 'g-s', markevery=12, label='Negativity')
plt.plot(ptbl, maxEigen, 'b--', label = 'Max eigenvalue')
plt.plot(ptbl, sumOthEigen, 'b:', label = 'Sum of oth. evs')
plt.plot(ptbl, conTbl, 'r:*', markevery=5, label='Concurrence')
plt.plot(ptbl, diffTbl, 'r:o',markevery=13, label='Difference')
#plt.plot(ptbl, diffTbl, linestyle=(0, (3, 5, 1, 5)), label='Difference')
plt.legend(bbox_to_anchor=(1.32, 1.0))
plt.axvline(0.33, color='gray',linestyle='--')
#plt.axhline(0.025, color='gray',linestyle='--')
plt.axvline(0.00, color='red',linestyle=':')
plt.text(0.175, -1.20,"Value of $p$",fontdict=font)
plt.text(-0.25, -0.90,"State is separable",fontdict=font2)
plt.text(-0.2,  -1.05,"$-0.33 \leq p \leq 0.33$",fontdict=font2)
plt.text(0.36,  -0.90,"State is entangled",fontdict=font2)
plt.text(0.45,  -1.05,"$p > 0.33$",fontdict=font2)
plt.savefig('sasd-isotropic-state.eps', bbox_inches='tight')
plt.show()

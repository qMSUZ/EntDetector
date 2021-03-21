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


from entdetector import *
import numpy as np

#
# Calculate the value of entropy of the density matrix
qden=create_x_two_qubit_random_state()
q0=entropy(qden)
print("Entroy = ",q0)

#
# Schmidt decomposition of Bell 1.0/sqrt{2} |00> + |11>
#
q = create_qubit_bell_state()
schmidt_shape=(2, 2)
s, e, f = schmidt_decomposition_for_vector_pure_state(q, schmidt_shape)
print("Number of schmidt coeff =", len(s))
print("Schmidt coeff =",s); print()

qrebuild = reconstruct_state_after_schmidt_decomposition(s,e,f)
print("State after rebuild =", qrebuild); print()
#
# Schmidt decomposition of random two pure qutrit state
#
qutrit=create_random_qudit_state(3, 2)
schmidt_shape=(3, 3)
s, e, f = schmidt_decomposition_for_vector_pure_state(qutrit, schmidt_shape)
print("Number of schmidt coeff =", len(s))
print("Schmidt coeff =",s); print()

#
# Schmidt decomposition of two pure qutrit state |00>
#
qutrit=create_pure_state(3, 2, 0)
schmidt_shape=(3, 3)
s, e, f = schmidt_decomposition_for_vector_pure_state(qutrit, schmidt_shape)
print("Number of schmidt coeff =", len(s))
print("Schmidt coeff =",s); print()


#
# Negativity for two-qubit W-State,
#

q = create_wstate(2)
qden = vector_state_to_density_matrix( q )

dim = 2
qdentmp = partial_tranpose(qden, [[dim,dim],[dim,dim]], [0, 1])
negativity_value  = (np.linalg.norm(qdentmp, 'nuc') - 1.0)/2.0
print("Negativity =", negativity_value); print()

print("Negativity with function negativity =", negativity(qden)); print()
print("Concurrence with function concurrence =", concurrence(qden)); print()

#
# Entropy 
#

typeOfState="Bell+"
#typeOfState="Bell-"
#typeOfState="W"

print("Type of state", typeOfState)
print()
d=2
p=-0.33
#p=0.0
step=0.01;
ptbl=[]
entTbl=[]
#negTbl=[]
#conTbl=[]
#diffTbl=[]
#maxEigen=[]
#sumOthEigen=[]
schmidt_shape=(2, 2)
while p <= (1.0 + step):
    #q = create_qubit_bell_state()
    #qdentmp = np.outer(q, q)
    #qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    qden = create_werner_two_qubit_state(p, typeOfState)
    e,s = create_sas_table_data(qden, schmidt_shape)
    #print("p=%.2f" % p)
    #pretty_matrix_print(qden)
    #print_sas_table( (e,s) )
    #(v0,v1,s0,s1) = calculate_statistic_for_sas_table(e,s)
    #print("p=%.2f" % p, "v=%.4f" % v0, "%.4f" % v1, "s=%.4f" % s0, "%.4f" % s1)
    entTbl.append(entropy(qden))
    print("for p=", p, "entropy=", entropy(qden))
    #negTbl.append(negativity(qden))
    #print("negativity=", negativity(qden))
    #conTbl.append(concurrence(qden))
    #print("concurrence=", concurrence(qden))
    #ptbl.append(p)
    #maxEigen.append( max([row[0] for row in e]) )
    #sumOthEigen.append( sum([row[0] for row in e]) - max([row[0] for row in e]) ) 
    #print("max eigen=", maxEigen[-1], "sum of other eigs=", sumOthEigen[-1])
    #print("difference between max eigen and sum oth evsl=", maxEigen[-1] - sumOthEigen[-1])
    #diffTbl.append(maxEigen[-1] - sumOthEigen[-1])
    p=p+step
    #print()
    #print()


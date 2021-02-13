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




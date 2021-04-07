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


print("Schmidt decomposition for Bell state")
print("expressed as pure state:")
print(create_qubit_bell_state())

qvec = create_qubit_bell_state()
s, u, vh = schmidt_decomposition_for_vector_pure_state(qvec, (2,2))
print("Schmidt cofficients = ", s)

qvec = create_qubit_bell_state()
r = schmidt_rank_for_vector_pure_state(qvec, (2,2))
print("Schmidt rank = ", r)

print()
print("Schmidt decomposition for Bell state")
print("expressed as density matrix:")
print(vector_state_to_density_matrix(create_qubit_bell_state()))

qvec = create_qubit_bell_state()
qden = vector_state_to_density_matrix( qvec )
s, e, f = schmidt_decomposition_operator(qden, (2,2))
print("Schmidt operator cofficients = ", s)

qvec = create_qubit_bell_state()
qden = vector_state_to_density_matrix( qvec )
s = schmidt_rank_for_operator(qden, (2,2))
print("Schmidt operator rank = ", s)



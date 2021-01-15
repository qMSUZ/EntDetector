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

from entdetector import *

q = create_qubit_bell_state()
schmidt_shp=(2, 2)
s,e,f = schmidt_decomposition_for_vector_pure_state(q, schmidt_shp)

qrebuild = reconstruct_state_after_schmidt_decomposition(s, e, f)
print(qrebuild)
print()

q0 = create_qubit_zero_state()
qplus = create_qubit_plus_state()
q = np.kron(q0, qplus)

schmidt_shape=(2, 2)
qden = vector_state_to_density_matrix( q )
sas_tbl = create_sas_table_data(qden, schmidt_shape)

print_sas_table( sas_tbl )
print()

q = create_qubit_bell_state()
qden = vector_state_to_density_matrix( q )
sas_tbl = create_sas_table_data(qden, schmidt_shape)

print_sas_table( sas_tbl )
print()







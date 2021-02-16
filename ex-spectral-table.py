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


q0 = create_qubit_plus_state()
qplus = create_qubit_one_state()

q = np.kron(q0, qplus)
qden = vector_state_to_density_matrix( q )
schmidt_shape=(2, 2)
sastbl = create_sas_table_data(qden, schmidt_shape)
print("SAS for states |0> and |+> ")
print_sas_table( sastbl )
print()


q = create_qubit_bell_state()
qden = vector_state_to_density_matrix( q )
schmidt_shape=(2, 2)
e,s = create_sas_table_data(qden, schmidt_shape)
print("SAS for states (1.0/sqrt(2)) (|00> + |11>) ")
print_sas_table( (e,s) )
print()


q = create_wstate( 3 )
qden = vector_state_to_density_matrix( q )
schmidt_shape=(2, 4)
e,s = create_sas_table_data(qden, schmidt_shape)
print("SAS for Werner three qubit state")
print_sas_table( (e,s) )
print()

q = create_ghz_state( 2, 3 )
qden = vector_state_to_density_matrix( q )
schmidt_shape=(2, 4)
e,s = create_sas_table_data(qden, schmidt_shape)
print("SAS for GHZ four three state")
print_sas_table( (e,s) )
print()

q = create_ghz_state( 2, 4 )
qden = vector_state_to_density_matrix( q )
schmidt_shape=(4, 4)
e,s = create_sas_table_data(qden, schmidt_shape)
print("SAS for GHZ four qubit state")
print_sas_table( (e,s) )
print()

q = create_ghz_state( 3, 3 )
qden = vector_state_to_density_matrix( q )
schmidt_shape=(3, 9)
e,s = create_sas_table_data(qden, schmidt_shape)
print("SAS for GHZ three qutrit state")
print_sas_table( (e,s) )
print()

qden = create_bes_horodecki_24_state(0.25)
schmidt_shape=(2, 4)
e,s = create_sas_table_data(qden, schmidt_shape)
print("SAS for Horodecki 2 and 4 state")
print_sas_table( (e,s) )
print()


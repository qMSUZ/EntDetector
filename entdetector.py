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

class ArgumentValueError(Exception):
    """ArgumentValueError"""
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
    """
        Create a base state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit that becomes zero state
        >>> q0=create_base_state(2, 1, 0)
        >>> print(q0)
        [1. 0.]
    """
    v = np.zeros(d ** n)
    v[base_state] = 1
    return v

def create_pure_state(d, n, base_state):
    """
        Create a pure state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of registry with one qubit that is a pure state
        >>> q0=create_pure_state(2, 1, 0)
        >>> print(q0)
        [1. 0.]
    """
    return create_base_state(d, n, base_state)

def create_qubit_zero_state():
    """
        Create a zero state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit for zero state
        >>> q0=create_qubit_zero_state()
        >>> print(q0)
        [1. 0.]
    """
    v = np.zeros(2)
    v[0] = 1.0
    return v

def create_qubit_one_state():
    """
        Create a one state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for one state qubit
        >>> q0=create_qubit_one_state()
        >>> print(q0)
        [0. 1.]
    """
    v = np.zeros(2)
    v[1] = 1.0
    return v

def create_qubit_plus_state():
    """
        Create a plus state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit which becomes plus state
        >>> q0=create_qubit_plus_state()
        >>> print(q0)
        [0.70710678 0.70710678]
    """
    v = np.zeros(2)
    v[0] = 1.0 / np.sqrt(2)
    v[1] = 1.0 / np.sqrt(2)
    return v

def create_qubit_minus_state():
    """
        Create a minus state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit which becomes minus state
        >>> q0=create_qubit_minus_state()
        >>> print(q0)
        [ 0.70710678 -0.70710678]
    """
    v = np.zeros(2)
    v[0] =   1.0 / np.sqrt(2)
    v[1] = - 1.0 / np.sqrt(2)
    return v

def create_qutrit_state(base_state):
    """
        Create a qutrit state

        Parameters
        ----------
        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for qutrit state
        >>> q0=create_qutrit_state(0)
        >>> print(q0)
        [1. 0. 0.]
    """
    v = np.zeros(3)
    v[base_state] = 1.0
    return v

def create_qutrit_zero_state():
    """
        Create a qutrit zero state

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for one qutrit which becomes zeros state
        >>> q0=create_qutrit_zero_state()
        >>> print(q0)
        [1. 0. 0.]
    """
    v = np.zeros(3)
    v[0] = 1.0
    return v

def create_qutrit_one_state():
    """
        Create a qutrit with state |1>

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of state |1> register for one qutrit
        >>> q0=create_qutrit_one_state()
        >>> print(q0)
        [0. 1. 0.]
    """
    v = np.zeros(3)
    v[1] = 1.0
    return v

def create_qutrit_two_state():
    """
        Create a qutrit with state |2>

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of state |2> register for one qutrit
        >>> q0=create_qutrit_two_state()
        >>> print(q0)
        [0. 0. 1.]
    """
    v = np.zeros(3)
    v[2] = 1.0
    return v

def create_qutrit_plus_state():
    """
        Create a qutrit plus state

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of qutrit plus state register for one qutrit
        >>> q0=create_qutrit_plus_state()
        >>> print(q0)
        [0.57735027 0.57735027 0.57735027]
    """
    v = np.ones(3)
    v[0] = 1.0/np.sqrt(3.0)
    v[1] = 1.0/np.sqrt(3.0)
    v[2] = 1.0/np.sqrt(3.0)
    return v

def create_qubit_bell_state(minus=0):
    """
        Create a qubit bell state

        Parameters
        ----------
            minus : integer 
                additional parameters for minus amplitude
        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of qubit bell state register for one qubit:
        >>> q0=create_qubit_bell_state()
        >>> print(q0)
        [0.70710678  0.         0.         0.70710678]

        Create of qubit bell state (with minus amplitude)
        register for one qubit:
        >>> q0=create_qubit_bell_state(1)
        >>> print(q0)
        [0.70710678  0.          0.         -0.70710678]

    """
    d = 2
    n = 2
    v = np.zeros(d ** n)
    v[0] = 1.0 / np.sqrt(2)
    if minus == 1:
        v[(d ** n) - 1] = -1.0 / np.sqrt(2)
    else:
        v[(d ** n) - 1] =  1.0 / np.sqrt(2)
    return v

def create_mixed_state(d,n):
    """
        Create a mixed state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        density matrix
            Numpy array for quantum state
            expressed as density matrix

        Examples
        --------
        Create of mixed state register for one qubit
        >>> q0=create_mixed_state(2, 1)
        >>> print(q0)
        [[0.5 0. ]
         [0.  0.5]]
    """
    qden = np.eye(d ** n) / (d ** n)
    return qden

#"""
#state |00..0> +  |kkk...k>
#where k = d - 1 and d is demension of single qudit of quantum register
#with n equally dimensional qudits
#"""
def create_0k_stat(d, n):
    """
        Create a 0K state (internal function)

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of |+> for one qubit
        >>> q0=create_0k_stat(2, 1)
        >>> print(q0)
        [0.70710678 0.70710678]

        Create of 1.0/sqrt(2.0)(|00> + |11>) for one qubit
        >>> q0=create_0k_stat(2, 1)
        >>> print(q0)
        [0.70710678 0.         0.         0.70710678]
    """
    v = np.zeros(d ** n)
    v[0] = 1.0/np.sqrt(2)
    v[-1] = v[0]
    return v

def create_max_entangled_pure_state(d):
    """
        Create a maximum entangled of pure state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for maximum entangled of pure state
        for two qubits:
        >>> q0=create_max_entangled_pure_state(2)
        >>> print(q0)
        [0.70710678 0.         0.         0.70710678]
    """
    v = np.reshape( np.eye(d), d**2 )
    v = v / np.sqrt( d )
    return v

def create_bes_horodecki_24_state(b):
    """
        Create a Horodecki's 2x4 of entangled state

        Parameters
        ----------
        b : real
            the entangled state with a parameter b

        Returns
        -------
        density matrix : numpy array
            Numpy array gives the Horodecki's two-qudit states

        Examples
        --------
        Create a Horodecki's 2x4 of entangled state
        >>> qden=create_bes_horodecki_24_state(1)
        >>> print(qden)
        [[0.125 0.    0.    0.    0.    0.125 0.    0.   ]
         [0.    0.125 0.    0.    0.    0.    0.125 0.   ]
         [0.    0.    0.125 0.    0.    0.    0.    0.125]
         [0.    0.    0.    0.125 0.    0.    0.    0.   ]
         [0.    0.    0.    0.    0.125 0.    0.    0.   ]
         [0.125 0.    0.    0.    0.    0.125 0.    0.   ]
         [0.    0.125 0.    0.    0.    0.    0.125 0.   ]
         [0.    0.    0.125 0.    0.    0.    0.    0.125]]
    """
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
    """
        Create a Horodecki's 3x3 of entangled state

        Parameters
        ----------
        a : real
            the entangled state with a parameter a

        Returns
        -------
        density matrix : numpy array
            Numpy array for the Horodecki's two-qutrit state
            expressed as density matrix

        Examples
        --------
        Create a Horodecki's 3x3 of entangled state
        >>> qden=create_bes_horodecki_33_state(1)
        >>> print(qden)
        [[0.11111111 0.         0.         0.         0.11111111 0.          0.         0.         0.11111111]
         [0.         0.11111111 0.         0.         0.         0.          0.         0.         0.        ]
         [0.         0.         0.11111111 0.         0.         0.          0.         0.         0.        ]
         [0.         0.         0.         0.11111111 0.         0.          0.         0.         0.        ]
         [0.11111111 0.         0.         0.         0.11111111 0.          0.         0.         0.11111111]
         [0.         0.         0.         0.         0.         0.11111111  0.         0.         0.        ]
         [0.         0.         0.         0.         0.         0.          0.11111111 0.         0.        ]
         [0.         0.         0.         0.         0.         0.          0.         0.11111111 0.        ]
         [0.11111111 0.         0.         0.         0.11111111 0.          0.         0.         0.11111111]]
    """
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
    """
        Create a GHz state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d,

        n : integer
            number of qudits for the created state

        Returns
        -------
        quantum state : numpy vector
            Numpy vector gives the d-partite GHZ state acting on local n dimensions

        Examples
        --------
        Create of register for a GHz state
        >>> q0=create_ghz_state(2, 2)
        >>> print(q0)
        [0.70710678 0.         0.         0.70710678]
    """
    g = np.zeros(d ** n)
    step = np.sum(np.power(d, range(n)))
    g[range(d) * step] = 1/np.sqrt(d)
    return g

def create_wstate(n):
    """
        Create a W-state

        Parameters
        ----------
        n : integer
            number of qudits for the created state

        Returns
        -------
        quantum state : numpy vector
            Numpy vector gives the n-qubit W-state

        Examples
        --------
        Create of register for a w state
        >>> q0=create_wstate(2)
        >>> print(q0)
        [0.         0.70710678 0.70710678 0.        ]
    """
    w = np.zeros(2 ** n)
    for i in range (n):
        w[2 ** i] = 1 / np.sqrt(n)
    return w

def create_isotropic_qubit_state(p):
    """
        Create a isotropic of qubit state
        Parameters
        ----------
        p : real
           The parameter of the isotropic state

        Returns
        -------
        density matrix : numpy array
           The isotropic state expressed
           as density matrix

        Examples
        --------
        Create of register for a isotropic of qubit state
        >>> q0=create_isotropic_qubit_state(0.25)
        >>> print(q0)
        [[0.3125 0.     0.     0.125 ]
         [0.     0.1875 0.     0.    ]
         [0.     0.     0.1875 0.    ]
         [0.125  0.     0.     0.3125]]
    """
    q = create_qubit_bell_state()
    qdentmp = np.outer(q, q)
    qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    return qden

def create_werner_two_qubit_state(p, state="Bell+"):
    """
        Create a Werner state for two qubit

        Parameters
        ----------
        p : real
           The parameter of the isotropic state
        state : string
           The name of quantum state: Bell+, Bell-, W.
           Default value is Bell+.

        Returns
        -------
        density matrix : numpy array
            The Werner state expressed
            as density matrix.

        Examples
        --------
        Create of register for two qubit to a Werner state
        between max entangled state and mixed state
        >>> q0=create_werner_two_qubit_state(0.25, state="Bell+")
        >>> print(q0)
        [[0.3125 0.     0.     0.125 ]
         [0.     0.1875 0.     0.    ]
         [0.     0.     0.1875 0.    ]
         [0.125  0.     0.     0.3125]]
    """
    if state=="Bell+":
        q = create_qubit_bell_state()
    if state=="Bell-":
        q = create_qubit_bell_state(minus=1)
    if state=="W":
        q = create_wstate(2)
    qdentmp = np.outer(q, q)
    qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    return qden

def create_chessboard_state(a,b,c,d,m,n):
    """
        Create a Chessboard state

        Parameters
        ----------
        a,b,c,d,m,n : integer
            The real arguments

        Returns
        -------
        density matrix
            Numpy array for quantum state
            expressed as density matrix


        Examples
        --------
        Create a Chessboard state
        >>> q0=create_chessboard_state(0.25, 0.5, 0.5, 0.1, 0.2, 0.8)
        >>> print(q0)
            [[ to fix  ]]


        """
    s = a * np.conj(c) / np.conj(n)
    t = a * d / m

    v1 = np.array([m, 0, s, 0, n, 0, 0, 0, 0])
    v2 = np.array([0, a, 0, b, 0, c, 0, 0, 0])
    v3 = np.array([(np.conj(n)), 0, 0, 0, (-np.conj(m)), 0, t, 0, 0])
    v4 = np.array([0, (np.conj(b)), 0, (-np.conj(a)), 0, 0, 0, d, 0])

    rho = np.outer(np.transpose(v1), v1) + np.outer(np.transpose(v2), v2) + np.outer(np.transpose(v3), v3) + np.outer(np.transpose(v4), v4)

    rho = rho/np.trace(rho)

    return rho

def create_gisin_state(lambdaX, theta):
    """
            Create a gisin state

            Parameters
            ----------
            lambdaX: float
                The real argument in 0 between 1 (closed interval)
            theta: float
                The real argument

        
            Returns
            -------
            density matrix
                Numpy array for Gisin state
                expressed as density matrix

            Examples
            --------
            Create a Gisin state
            >>> q0=create_gisin_state(0.25, 2)
            >>> print(q0)            
                [[0.375      0.         0.         0.        ]
                 [0.         0.20670545 0.09460031 0.        ]
                 [0.         0.09460031 0.04329455 0.        ]
                 [0.         0.         0.         0.375     ]]
    """
    rho_theta = np.array([[0, 0, 0, 0],
                [0, (np.sin(theta) ** 2), (-np.sin(2 * theta) / 2), 0],
                [0, (-np.sin(2 * theta) / 2), (np.cos(theta) ** 2), 0],
                [0, 0, 0, 0]])

    rho_uu_dd =np.array(
                [[1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]])

    gisin_state = lambdaX * rho_theta + (1- lambdaX) * rho_uu_dd / 2

    return gisin_state

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
    """
        Create a eigen decomposition

        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix

        Returns
        -------
        eigval : numpy array
        eigvec : numpy array
            The vector and array of a eigenvalues and eigenvectors

        Examples
        --------
        Create eigen decomposition of given quantum state:
        >>> qden=create_werner_two_qubit_state(0.75)
        >>> ed=eigen_decomposition(qden)
        >>> print(ed)
        (array([0.0625, 0.0625, 0.0625, 0.8125]), array([[-0.70710678,  0.        ,  0.        , -0.70710678],
               [ 0.        ,  0.        , -1.        ,  0.        ],
               [ 0.        ,  1.        ,  0.        ,  0.        ],
               [ 0.70710678,  0.        ,  0.        , -0.70710678]]))
    """
    eigval, eigvec = np.linalg.eigh(qden)
    return eigval, eigvec

def eigen_decomposition_for_pure_state(q):
    """
        Create a eigen decomposition for pure state

        Parameters
        ----------
        q : numpy vector
            The parameter q represents the vector state.
            The input vector is converted to density matrix.

        Returns
        -------
        A two element tuple (eigval,eigvec) where:
            eigval : is a numpy array of a eigenvalues,
            eigvec : is a numpy array of a eigenvectors.

        Examples
        --------
        Create of register for eigen decomposition for pure state
        >>> q = create_qubit_bell_state()
        >>> ed=eigen_decomposition_for_pure_state(q)
        >>> print(ed)
        (array([0., 0., 0., 1.]), array([[-0.70710678,  0.        ,  0.        , -0.70710678],
               [ 0.        ,  0.        , -1.        ,  0.        ],
               [ 0.        ,  1.        ,  0.        ,  0.        ],
               [ 0.70710678,  0.        ,  0.        , -0.70710678]]))
    """
    qden = np.outer(q,q)
    eigval,eigvec = np.linalg.eigh(qden)
    return eigval, eigvec

def reconstruct_density_matrix_from_eigen_decomposition(eigval, eigvec):
    """
        Reconstruction of density matrix from a eigen decomposition

        Parameters
        ----------
        eigval : numpy array
        eigvec : numpy array
            The vector and array of a eigenvalues and eigenvectors

        Returns
        -------
        density matrix : numpy array
            Numpy array for reconstructed quantum state

        Examples
        --------
        Reconstruction of density matrix from eigen decomposition:
        >>> q = create_qubit_bell_state()
        >>> qden = vector_state_to_density_matrix(q)
        >>> ev,evec = eigen_decomposition(qden)
        >>> qdenrecon = reconstruct_density_matrix_from_eigen_decomposition(ev, evec)
        >>> print( qdenrecon )
        [[0.5 0.  0.  0.5]
         [0.  0.  0.  0. ]
         [0.  0.  0.  0. ]
         [0.5 0.  0.  0.5]]
     """

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
    """
        Create a Schmidt decomposition for vector pure state

        Parameters
        ----------
        q : numpy vector
            The parameter q represents the vector state

        decomposition_shape : tuple of two integers
            Dimensions of two subsystems

        Returns
        -------
        A three element tuple (s,u, vh) where:
           s  : numpy vector containing Schmidt coefficients,
           u  : arrays of left Schmidt vectors,
           vh : arrays of right Schmidt vectors.

        Examples
        --------
        Create of register to Schmidt decomposition for vector pure state
        >>> q = create_qubit_bell_state()
        >>> decomposition_shape=(2, 2)
        >>> sd=schmidt_decomposition_for_vector_pure_state(q, decomposition_shape)
        >>> print(sd)
        (array([0.70710678, 0.70710678]), array([[1., 0.],
               [0., 1.]]), array([[1., 0.],
               [0., 1.]]))
    """
    d1,d2 = decomposition_shape
    m = q.reshape(d1, d2)
    u, s, vh = np.linalg.svd(m, full_matrices=True)
    
    return s, u, vh

def schmidt_decomposition_for_square_operator(qden, decomposition_shape):
    pass

def schmidt_rank_for_vector_pure_state(q, decomposition_shape):
    """
        Calculate a Schmidt rank for vector pure state

        Parameters
        ----------
        q : numpy array
            The parameter of the vector state

        decomposition_shape : tuple of two integers
            Dimensions of two subsystems

        Returns
        -------
        sch_rank : integer
            Schmidt rank value as integer number

        Examples
        --------
        Calculate of Schmidt rank for vector pure state
        >>> q = create_qubit_bell_state()
        >>> decomposition_shape=(2, 2)
        >>> sr=schmidt_rank_for_vector_pure_state(q, decomposition_shape)
        >>> print(sr)
            2
    """
    d1,d2 = decomposition_shape
    m = q.reshape(d1, d2)
    sch_rank = np.linalg.matrix_rank(m)
    return sch_rank

def reconstruct_state_after_schmidt_decomposition(s, e, f):
    """
        Reconstruction state after Schmidt decomposition

        Parameters
        ----------
        s : numpy array
            The values of Schmidt coefficients

        e, f : numpy arrays
            The basis vectors from Schmidt decomposition

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Reconstruction state after Schmidt decomposition:
        >>> q = create_qubit_bell_state()
        >>> schmidt_shp=(2, 2)
        >>> s,e,f = schmidt_decomposition_for_vector_pure_state(q,schmidt_shp)
        >>> q0=reconstruct_state_after_schmidt_decomposition(s, e, f)
        >>> print(q0)
            [0.70710678 0.         0.         0.70710678]
    """

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
    """
        Computes the entropy of a density matrix
        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix
        logbase : string
            A string represents the base of the logarithm: 
               "e", "2", and "10".

        Returns
        -------
        entropy_val : float
            The value of entropy
        Examples
        --------
        Calculate the value of entropy of the density matrix
        >>> qden=create_x_two_qubit_random_state()
        >>> q0=entropy(qden, "10")
        >>> print(q0)
            0.5149569745101069
    """
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
    """
        Computes a negativity of bipartite density matrix
        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix
        d : integer
            the number of degrees of freedom for the qudit d,
        n : integer
            number of qudits for the created state

        Returns
        -------
        negativity_value : float
            The value of negativity of bipartite density matrix

        Examples
        --------
        Calculate the value for a negativity of bipartite density matrix
        >>> q = create_wstate(d)
        >>> qden = vector_state_to_density_matrix( q )
        >>> q0=negativity(qden)
        >>> print(q0)
            0.4999999999999998
    """
    if np.isscalar(qden):
        raise DimensionError("Wrong dimension of argument!")
        return None
    dim = int(np.log(d ** n)/np.log(d))
    qdentmp = partial_tranpose(qden, [[dim,dim], [dim,dim]], [0, 1])
    negativity_value = (np.linalg.norm(qdentmp, 'nuc') - 1.0)/2.0
    return negativity_value

#
# Concurrence
#

def concurrence( qden ):
    """
        Computes value a concurrence for a two-qubit state
        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix
        Returns
        -------
        c : float
            The value of concurrence of two-qubit state

        Examples
        --------
        Calculate the value of concurrence for a two-qubit state
        >>> qden=create_werner_two_qubit_state(0.79)
        >>> q0=concurrence(qden)
        >>> print(q0)
            0.6849999999999994
    """
    if np.isscalar(qden):
        raise DimensionError("Wrong dimension of argument!")
        return None
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

def make_partititon_as_list(kappa):
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

def filtered_data_for_paritition_division( r, idxtoremove ):
    rr = [ ]
    schmidtnumbers = [ ]
    #
    #  remove qubits that are not entangled
    #
    for i in r:
        for idx in idxtoremove:
            if idx in i[1][0]:
                i[1][0].remove(idx)
            #end if
            if idx in i[1][1]:
                i[1][1].remove(idx)
            #end if
        #end for
        #if len(i[1][0])>1 and len(i[1][1])>1:
        rr=rr+[i]
        if i[0] not in schmidtnumbers:
            schmidtnumbers=schmidtnumbers + [i[0]]
        #end if
    #end for
    
    # print 'schmidt numbers', schmidtnumbers
    #
    # sort by Schmidt rank
    #
    rr=sorted(rr)
    print("sorted partitions")
    for i in rr:
        print(i)
    #end for
    #
    # building a set of partitions
    #
    finalpart=set()
    if 1 in schmidtnumbers:
        for i in rr:
            if i[0]==1:
                if len(i[1][0])>1:
                    finalpart.add(tuple(i[1][0]))
                #end of if
                if len(i[1][1])>1:
                    finalpart.add(tuple(i[1][1]))
                #end of if
            #end of if
        #end of for i in rr
    #end of if
    
    if 1 not in schmidtnumbers:
        finalcluster=[]
        for i in rr:
            if i[0]==2:
                for e in i[1][0]:
                    if e not in finalcluster:
                        finalcluster = finalcluster + [ e ]
                for e in i[1][1]:
                    if e not in finalcluster:
                        finalcluster = finalcluster + [ e ]
        finalpart.add(tuple(finalcluster))
        #print('final cluster', finalcluster)
        #print('final part', finalpart)
    return finalpart

def bin2dec(s):
    return int(s, 2)

def ent_detection_by_paritition_division( q, nqubits, verbose = 0 ):
    #s = q.size
    s = nqubits
    # generation of all two partitite divisions of given
    # set which is build from the quantum register q
    res = [ ]
    idxtoremove = [ ]
    k = [0] * s
    M = [0] * s
    p = 2
    partititon_p_initialize_first(k, M, p)
    lp = []
    lp = lp + [make_partititon_as_list(k)]
    while partition_p_next(k, M, p):
        lp = lp + [make_partititon_as_list(k)]
    for i in lp:
            if verbose==1:
                    print(i[0], i[1])
            mxv=2**len(i[0])
            myv=2**len(i[1])
            if verbose:
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
                            if verbose:
                                    print("D("+xstr+","+ystr+")","D(",dxidx,dyidx,") C",dcidx,cidx,cstr)
                            #m.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            m[dxidx,dyidx] = q[dcidx]
                            #mt.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            mt[dxidx,dyidx] = q[dcidx]
            if verbose:
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

            if verbose:
                    print("non zero:", ev_count)
            if (ev_count==1) and (len(i[0])==1):
                idxtoremove=idxtoremove+i[0]
                
            if (ev_count==1) and (len(i[1])==1):
                idxtoremove=idxtoremove+i[1]
                
            res=res + [[ev_count, [i[0], i[1]]]]
    return res,idxtoremove

def detection_entanglement_by_paritition_division( q, nqubits, verbose = 0 ):
    [r,idxtoremove]=ent_detection_by_paritition_division( q, nqubits, verbose )
    print("idx to remove", idxtoremove)
    print("all partitions")
    for i in r:
        print(i)
    fp = filtered_data_for_paritition_division( r, idxtoremove )
    if len(fp)==0:
        print("register is fully separable")
    else:
        print("raw final filtered data")
        for i in fp:
            print(i)
    
    cfp = set(fp)
    ffp = set(fp)
    
    for i in fp:
        if i in cfp:
            cfp.remove(i)
        for e in cfp:
            if (set(i) < set(e)) and (len(i)!=len(e)):
                if e in ffp:
                    ffp.remove(e)
    print("final filtered data")
    for i in ffp:
        print(i)




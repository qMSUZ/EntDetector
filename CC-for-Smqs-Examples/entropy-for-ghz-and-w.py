#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2021 -- 2022 by Marek Sawerwain                         *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                              by Joanna Wiśniewska                       *
# *                                  <Joanna.Wisniewska@wat.edu.pl>         *
# *                                                                         *
# *                              by Roman Gielerak                          *
# *                                  <R.Gielerak@issi.uz.zgora.pl>          *
# *                                                                         *
# * file is related to the paper titled                                     *
# *   Classical Computer assisted  analysis  of small multi-qudit systems   *
# *    published in IEEE Access Vol.???, No.???, pages ?????, Year          *
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

#import math
import mpmath as mpm

import matplotlib as mpl
import matplotlib.pyplot as plt
#import numpy as np
#import contextlib


mpm.mp.dps = 200;

def calcEN(N, d):
    Nstar=mpm.floor(N/mpm.mpf('2.0'))
    tmp=mpm.mpf('0.0')
    k=mpm.mpf('1.0')
    while k <= Nstar:
        tmp=tmp+k*mpm.binomial(N, k)
        k=k+mpm.mpf('1.0')
    return mpm.log(mpm.power(d, tmp))


def calcEGHZN2(N, d):
    Nstar=mpm.floor(N/mpm.mpf('2.0'))
    tmp=mpm.mpf('0.0')
    k=mpm.mpf('1.0')
    while k <= Nstar:
        tmp = tmp + (mpm.binomial(N, k)*mpm.log(d))
        k=k+mpm.mpf('1.0')
    return tmp

def calcE1kWN2(N,k):
    #return -1.0*(1.0-1.0/N)*math.log(1.0-1.0/N) - k*(1.0/N)*math.log(1.0/N)
    return mpm.mpf('-1.0')*(mpm.mpf('1.0')-(mpm.mpf('1.0')/N))*mpm.log(mpm.mpf('1.0')-mpm.mpf('1.0')/N) - k*(mpm.mpf('1.0')/N)*mpm.log(mpm.mpf('1.0')/N)

def calcEWN2(N):
    Nstar=mpm.floor(N/mpm.mpf('2.0'))
    tmp=mpm.mpf('0.0')
    k=1
    while k <= Nstar:
        tmp=tmp+mpm.binomial(N, k)*calcE1kWN2(N,k)
        k=k+mpm.mpf('1.0')
    return tmp
  
 

def prepare_figures(Nrange, d):
    xv=[]
    
    yv1=[]
    yv1P1=[]
    yv1P2=[]
    
    yv2GHZ=[]
    yv2GHZP1=[]
    yv2GHZP2=[]
    
    yy2Diff=[]
    yy2DiffP1=[]
    yy2DiffP2=[]
    yv3W=[]
    yy3Diff=[]
    
    for nidx in range(1,Nrange+1):
        N=nidx
        ENd=calcEN(N,d)
        ENdP1=calcEN(N,d+1)
        ENdP2=calcEN(N,d+2)
        EGHZN2=calcEGHZN2(N, d)
        EGHZN2P1=calcEGHZN2(N, d+1)
        EGHZN2P2=calcEGHZN2(N, d+2)
        EWN2=calcEWN2(N)
        #print('ENd', ENd, 'EGHZN2=', EGHZN2, 'EWN2=', EWN2)
        xv.append(nidx)
        yv1.append(ENd)
        yv1P1.append(ENdP1)
        yv1P2.append(ENdP2)
        yv2GHZ.append(EGHZN2)
        yv2GHZP1.append(EGHZN2P1)
        yv2GHZP2.append(EGHZN2P2)
        yy2Diff.append(ENd-EGHZN2)
        yy2DiffP1.append(ENdP1-EGHZN2P1)
        yy2DiffP2.append(ENdP2-EGHZN2P2)
        yv3W.append(EWN2)
        yy3Diff.append(ENd-EWN2)
        
    plt.cla()
    plt.clf()
       
    plt.plot(xv, yv1P2, 'r:', label='Total Entropy for d='+str(d+2))
    plt.plot(xv, yy2DiffP2, 'b:', label='Deficit of entanglement for d='+str(d+2))

    plt.plot(xv, yv1P1, 'r-.', label='Total Entropy for d='+str(d+1))
    plt.plot(xv, yy2DiffP1, 'b-.', label='Deficit of entanglement for d='+str(d+1))

    plt.plot(xv, yv1, 'r.-', label='Total Entropy for d='+str(d))
    plt.plot(xv, yy2Diff, 'b.-', label='Deficit of entanglement for d='+str(d))

    titleplot_for_GHZ='Entropy for $\mathrm{GHZ}_N('+str(d+2)+')$';
    plt.plot(xv, yv2GHZP2, 'g:', label=titleplot_for_GHZ)
    titleplot_for_GHZ='Entropy for $\mathrm{GHZ}_N('+str(d+1)+')$';
    plt.plot(xv, yv2GHZP1, 'g-.', label=titleplot_for_GHZ)
    titleplot_for_GHZ='Entropy for $\mathrm{GHZ}_N('+str(d)+')$';
    plt.plot(xv, yv2GHZ, 'g.-', label=titleplot_for_GHZ)

    plt.yscale('log')
    mpl.pyplot.ylim(4,10e4)
    
    plt.grid()
    
    plt.legend(bbox_to_anchor=(0.65,0.55))
    #plt.text(0.175, -1.20,"Value of $p$",fontdict=font)
    plt.xlabel("Value of N")
    plt.ylabel("Value of Entropy")
    
    figure_file_name='entropy-for-ghz-n-{0}.eps'.format(d)
    plt.savefig(figure_file_name, bbox_inches='tight')
    plt.show()
    
    plt.plot(xv, yv1, 'r.-', label='Total Entropy for d='+str(d))
    titleplot_for_W='Entropy for $\mathrm{W}_N('+str(d)+')$';
    plt.plot(xv, yv3W, 'g.-', label=titleplot_for_W)
    plt.plot(xv, yy3Diff, 'b.-', label='Deficit of entanglement')
    plt.legend(bbox_to_anchor=(0.5,0.95))
    #plt.text(0.175, -1.20,"Value of $p$",fontdict=font)
    plt.xlabel("Value of N")
    plt.ylabel("Value of Entropy")
    
    plt.yscale('log')
    mpl.pyplot.ylim(4,10e4)
    
    plt.grid()
    
    figure_file_name='entropy-for-w-n-{0}.eps'.format(d)
    plt.savefig(figure_file_name, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    prepare_figures(12, 2)


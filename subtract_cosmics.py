#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 08:52:51 2020

@author: korepashki
"""

import numpy as np

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300 # default resolution of the plot
import matplotlib.pyplot as plt
plt.style.use('bmh')

from expspec import *
from fit_single_peak import *


def subtract_cosmic_spikes_moll(x, y, width=0, display=2):
    """
    width : width of cosmic spikes, in units of x-axis,
                                    not in pixels!
        By default width = distance be two pixels of CCD;
            if the data were automatically processed (like, say, in Bruker Senterra),
            then the width should be set to ~2
    The script assumes that x are sorted in the ascending order.
    """
    
    if width==0:
        width = np.abs(x[1]-x[0])
        if display > 0:
            print (' [auto]width = ', width)
        width_in_pixels = 3

    else:
        width_in_pixels = width * len(x) / np.abs(x[-1]-x[0])
        if display > 0:
            print(' width in pixels = ', width_in_pixels)

    # calculate the mollification_width:    
    mollification_width = int(2 * np.ceil(width_in_pixels) + 1)
    
    if display > 0:
        print (' mollification_width = ', mollification_width, 'pixels') # , 'type:', type(mollification_width ))
    
    derspec_current = ExpSpec(x, y)

    current_peak_position = np.inf
    iteration_number = 0
    while True:
        print('\niteration number ', iteration_number)
        y_moll = moving_average_molification(derspec_current.y,
                                    struct_el=mollification_width)

        y_modscore = np.abs(derspec_current.y - y_moll)
        if display > 0:
            plt.plot(x, y_modscore/np.max(y_modscore), 'r', linewidth=0.5)
            plt.plot(x, (derspec_current.y-np.min(derspec_current.y))/(np.max(derspec_current.y)-np.min(derspec_current.y)), 'k', linewidth=1)
            plt.text(0.16, 0.92, 'current spectrum', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, color='k')
            plt.text(0.16, 0.82, 'detection score', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, color='r')
            plt.tick_params(labelleft=False)
            plt.title('iteration ' + str(iteration_number))
            plt.show()
        iteration_number += 1
        # find largest peak:
        top_peak_index = (np.abs(y_modscore)).argmax()
        if display > 1:
            print('\n top score index = ', top_peak_index)
        cosmic_position = x[top_peak_index]
        peaksign = np.sign(derspec_current.y[top_peak_index] - y_moll[top_peak_index])

        if cosmic_position == current_peak_position:
            # same peak again, break
            if display > 0:
                print('spectrum done')
            break
        else:
            current_peak_position = cosmic_position


        if display > 0:
            print(' cosmic peak suspect at x = ', cosmic_position, ', sign =', peaksign, '; \n now fitting:')

        # fit:
        cosmic_spike = fit_single_peak(derspec_current,
                                       peak_position = cosmic_position,
                                        fitrange=(cosmic_position-8*width, cosmic_position+8*width),
                                        fwhm=width,
                                        display=1,
                                        peaksign=peaksign)
            # subtract:
        peak2subtract = CalcPeak(x)
        peak2subtract.specs_array = cosmic_spike.specs_array
        if peak2subtract.fwhm <= width:
            if display > 0:
                print('fwhm of the fitted peak = ', peak2subtract.fwhm)
                # print('threshold width = ', width)
            derspec_current = ExpSpec(x, derspec_current.y-peak2subtract.curve)
        else:
            if display > 0:
                print('\nthe current peak is broad, stopping')
            break

    if display > 0:
        plt.plot(x, y, 'k', linewidth=1)
        plt.plot(x, derspec_current.y, 'r', linewidth=1)
        plt.text(0.16, 0.92, 'raw', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, color='k')
        plt.text(0.16, 0.82, 'de-spiked', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, color='r')
        plt.title('final')
        plt.show()

    return derspec_current.y


if __name__ == '__main__':

    s2test = np.genfromtxt('test_data/graphene_2x240s.txt')
    wn = s2test[:,0]
    graphene_w_cosmics = s2test[:,1]
    y_sub = subtract_cosmic_spikes_moll(wn, graphene_w_cosmics, width=3.2, display=1)
    


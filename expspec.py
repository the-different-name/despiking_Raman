#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:06:53 2020

@author: korepashki

"""

import numpy as np


class ExpSpec ():
    def __init__(self, full_x, full_y) :
        self.full_x = full_x
        self.full_y = full_y
        self.xrange = (np.min(full_x), np.max(full_x))
        self.x = full_x
        self.y = full_y
        
    @property
    def working_range(self):
        return self.xrange
    @working_range.setter
    def working_range(self, xrange):
        self.xrange = (np.maximum(np.min(xrange), np.min(self.full_x)), np.minimum(np.max(xrange), np.max(self.full_x)))
        self.x = self.full_x[np.where(np.logical_and(self.full_x>=np.amin(xrange), self.full_x<=np.amax(xrange)))]
        self.y = self.full_y[np.where(np.logical_and(self.full_x>=np.amin(xrange), self.full_x<=np.amax(xrange)))]
    

class SpectralFeature () :
    """ Abstract spectral feature, with no x-axis defined
     Order of parameters in array:
     0:    x0 (default 0)
     1:    fwhm (defauld 1)
     2:    asymmetry (default 0)
     3:    Gaussian_share (default 0, i.e. Lorentzian peak)
     4:    voigt_amplitude (~area, not height)
     5:    Baseline slope (k) for linear BL
     6:    Baseline offset (b) for linear BL
    """

    def __init__(self) :
        self.specs_array = np.zeros(7)
        self.specs_array[1] = 1 # set default fwhm to 1. Otherwise we can get division by 0

    @property
    def position(self):
        return self.specs_array[0]
    @position.setter
    def position (self, position) :
        self.specs_array[0] = position

    @property
    def fwhm(self):
        return self.specs_array[1]
    @fwhm.setter
    def fwhm (self, fwhm) :
        self.specs_array[1] = fwhm
    
    @property
    def asymmetry(self):
        return self.specs_array[2]
    @asymmetry.setter
    def asymmetry (self, asymmetry) :
        self.specs_array[2] = asymmetry

    @property
    def Gaussian_share(self):
        return self.specs_array[3]
    @Gaussian_share.setter
    def Gaussian_share (self, Gaussian_share) :
        self.specs_array[3] = Gaussian_share

    @property
    def voigt_amplitude(self):
        return self.specs_array[4]
    @voigt_amplitude.setter
    def voigt_amplitude (self, voigt_amplitude) :
        self.specs_array[4] = voigt_amplitude

    @property
    def BL_slope(self):
        return self.specs_array[5]
    @BL_slope.setter
    def BL_slope (self, BL_slope) :
        self.specs_array[5] = BL_slope

    @property
    def BL_offset(self):
        return self.specs_array[6]
    @BL_offset.setter
    def BL_offset (self, BL_offset) :
        self.specs_array[6] = BL_offset


class CalcPeak (SpectralFeature) :
    """ Asymmetric peak calculated on x-asis (a grid of wavenumbers).
    It is possible to set a peak height,
        Changing fwhm keeps area same, while changes height.
        Changing height changes area while keeps fwhm.
    """

    def __init__(self, wn=np.linspace(0, 1, 129)) :
        super().__init__()
        self.wn = wn
        self.specs_array[0] = (wn[-1]-wn[0])/2

    @property
    def peak_area (self) :
        peak_area = (1 - self.specs_array[3]) * self.specs_array[4] * (1 + 0.69*self.specs_array[2]**2 + 1.35 * self.specs_array[2]**4) + self.specs_array[3] * self.specs_array[4] * (1 + 0.67*self.specs_array[2]**2 + 3.43*self.specs_array[2]**4)
        return peak_area

    @property
    def peak_height (self) :
        amplitudes_L = self.specs_array[4]*2/(np.pi*self.specs_array[0])
        amplitudes_G = self.specs_array[4]*(4*np.log(2)/np.pi)**0.5 / self.specs_array[0]
        peak_height = self.specs_array[3] * amplitudes_G + (1-self.specs_array[3]) * amplitudes_L
        return peak_height
    @peak_height.setter
    def peak_height(self, newheight):
        self.specs_array[4] = newheight / (
                                self.specs_array[3] * (4*np.log(2)/np.pi)**0.5 / self.specs_array[1] + (1-self.specs_array[3]) * 2/(np.pi*self.specs_array[1])
                                )

    @property
    def fwhm_asym (self) :
        """ real fwhm of an asymmetric peak"""
        fwhm_asym = self.fwhm * (1 + 0.4*self.asymmetry**2 + 1.35*self.asymmetry**4)
        return fwhm_asym
    
    @property
    def curve (self) :
        """ Asymmetric pseudo-Voigt funtion as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = self.specs_array[4] * voigt_asym(self.wn-self.specs_array[0], self.specs_array[1], self.specs_array[2], self.specs_array[3])
        return curve

    @property
    def curve_with_BL (self):
        curve_with_BL = self.curve + self.specs_array[5] * (self.wn - self.specs_array[0]) + self.specs_array[6]
        return curve_with_BL

def moving_average_molification (rawspectrum, struct_el=7):
    """Moving average mollification
    """

    molifier_kernel = np.ones(struct_el)/struct_el
    denominormtor = np.convolve(np.ones_like(rawspectrum), molifier_kernel, 'same')
    smoothline = rawspectrum

    smoothline = np.convolve(smoothline, molifier_kernel, 'same') / denominormtor

    return smoothline



def voigt_asym(x, fwhm, asymmetry, Gausian_share):
    """ returns pseudo-voigt profile composed of Gaussian and Lorentzian,
         which would be normalized by unit area if symmetric
         The funtion as defined in Analyst: 10.1039/C8AN00710A"""
    x_distorted = x*(1 - np.exp(-(x)**2/(2*(2*fwhm)**2))*asymmetry*x/fwhm)
    Lor_asym = fwhm / (x_distorted**2+fwhm**2/4) / (2*np.pi)
    Gauss_asym = (4*np.log(2)/np.pi)**0.5/fwhm * np.exp(-(x_distorted**2*4*np.log(2))/fwhm**2)
    voigt_asym = (1-Gausian_share)*Lor_asym + Gausian_share*Gauss_asym
    return voigt_asym


if __name__ == '__main__':
    print('This script contains supplementary operations')
"""
fit single peak with optional parameters
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import detrend
from expspec import CalcPeak, voigt_asym
import matplotlib.pyplot as plt


def fit_single_peak(derspec, peak_position=None, fitrange=None, peaksign=0, fwhm=None, exp4sign=0.1, display=0):
    """ Returns class calcpeak
     derspec: class expspec
     peak_position: optional starting value.
     fitrange: (x1, x2) is the fitting range
     peaksign = 1 or -1 for positive or negative peaks
     fwhm by default is 4 inter-point distances (starting value)
     exp4sign is a setting for fit asymmetry
     Display = 0, 1 or 2 for no print-out, final print-out or detailed print-out
     Display = 2 plots the fit results
     """

    # step 0: initialize the calcpeak:
    derpeak = CalcPeak(derspec.x)
    # capture the original working range, which has to be restored later:
    original_working_range = derspec.working_range
    
    # step 1: check if we need to set x0 and find the index of x0
    if fitrange != None: 
        derspec.working_range = fitrange
        #@Test&Debug # 
        if display > 1:
            print('fitting range from input: ', derspec.working_range)
    if peak_position == None:
        # find index of a maximum y:
        if peaksign != 0:
            idx0 = (peaksign * detrend(derspec.y)).argmax()
        else:
            idx0 = (np.abs(detrend(derspec.y))).argmax()
            peaksign = np.sign((detrend(derspec.y))[idx0])
            if display > 1:
                print('peak sign detected as ', peaksign)
        derpeak.position = derspec.x[idx0]
        #@Test&Debug 
        if display > 1:
            print('x0 set by default to ', derpeak.position)
    else : 
        derpeak.position = peak_position
        if peaksign == 0:
            idx0 = (np.abs(derspec.x - peak_position)).argmin()
            peaksign = np.sign((detrend(derspec.y))[idx0])
            if display > 1:
                print('peak sign detected as ', peaksign)
            
        #@Test&Debug 
        if display > 1:
            print('x0 set from input:', derpeak.position)

    # step 2: set initial value of fwhm from input or to 5 points (4 inter-point distances)
    if fwhm == None:
        interpoint_distance = (derspec.x[-1]-derspec.x[0]) / (len(derspec.x)-1)
        derpeak.fwhm = abs(4*interpoint_distance)
        #@Test&Debug 
        if display > 1:
            print ('fwhm is set to 5 points: {} cm\N{superscript minus}\N{superscript one}'.format(derpeak.fwhm))
    else:
        derpeak.fwhm = fwhm
        if display > 1:
            print ('fwhm is set from input: {} cm\N{superscript minus}\N{superscript one}'.format(derpeak.fwhm))
    
    # step 3: Set initial working range
    if fitrange == None: # set to +-4*fwhm
        idx0 = (np.abs(derspec.x-derpeak.position)).argmin() # find point number for a closest to x0 point:
        #@Test&Debug # print('x0 is point number ', idx0)
        #@Test&Debug # 
        if display > 1:
            print('fitting range by default: ', derspec.working_range)
        derspec.working_range = (derspec.x[idx0]-4*derpeak.fwhm, derspec.x[idx0]+4*derpeak.fwhm)

    # step 4: Set fitting range
    derpeak.wn = derspec.x

    # step 5: Set other starting values
    derpeak.voigt_amplitude = 0
    derpeak.asymmetry = 0
    startingpoint = np.zeros(7)
    bounds_high = np.full_like(startingpoint, np.inf)
    bounds_low = np.full_like(startingpoint, -np.inf)
    bounds_low[2] = -0.36; bounds_high[2] = 0.36 # asymmetry
    bounds_low[3] = 0; bounds_high[3] = 1 # Gaussian share
    
    while True :
        # 1: find index of a y_max, y_min within the fitting range, 
        idx0local = (np.abs(derspec.x - derpeak.position)).argmin()
        # find most deviant from y0 point:
        #   and also peak height
        # if peaksign == 0:
        #     peak_height = (np.abs((detrend(derspec.y))[idx0local]))
        # else:
        peak_height = peaksign * (np.abs((detrend(derspec.y))[idx0local]))
        
        ymin_local = derspec.y[idx0local] - peak_height
        #@Test&Debug # 
        if display > 1:
            print('starting peak position =', derspec.x[idx0local], ', peak_height = ', peak_height)
            print('starting FWHM =', derpeak.fwhm, ', fitting range = ', derspec.working_range)

        derpeak.peak_height = peak_height
        startingpoint = derpeak.specs_array
        startingpoint[5] = 0 # always start next round with the flat baseline
        startingpoint[6] = ymin_local

        # 2: set bounds for parameters:
         # position: x0 +- 2 * fwhm (0)
        bounds_low[0] = derpeak.position - np.sign(derspec.x[-1]-derspec.x[0]) * derpeak.fwhm * 2
        bounds_high[0] = derpeak.position + np.sign(derspec.x[-1]-derspec.x[0]) * derpeak.fwhm * 2
         # fwhm: 0.25-8x fwhm (0)
        bounds_low[1] = 0.25*derpeak.fwhm
        if fwhm == None:
            bounds_high[1] = abs((derspec.x[-1]-derspec.x[0]))/2
        else:
            bounds_high[1] = 8*derpeak.fwhm
         # amplitude depending on sign:
        if peaksign > 0:
            bounds_low[4] = 0
        elif peaksign < 0:
            bounds_high[4] = 0


        def func2min(peakparams): # x0, fwhm, asymmetry, Gausian_share, voigt_amplitude, baseline_k, baseline_y0):
            thediff = derspec.y - (
                peakparams[4] * voigt_asym(derspec.x-peakparams[0], peakparams[1], peakparams[2], peakparams[3]) +
                peakparams[5] * (derspec.x-peakparams[0]) + peakparams[6] )
            derfunc = (thediff * 
                       np.exp(exp4sign*(1 - peaksign*np.sign(thediff))**2)
                        )
            return derfunc
        
        try:
            solution = least_squares(func2min, startingpoint, bounds=[bounds_low, bounds_high], verbose=0) # , max_nfev=1e4)
            converged_parameters_linear_bl = solution.x
            derpeak.specs_array = converged_parameters_linear_bl
            #     #@Test&Debug #
            if display > 1:
                print('least_squares converged')
            break
        except RuntimeError: # (RuntimeError, TypeError, NameError):
            #     #@Test&Debug #
            if display > 1:
                print('least_squares optimization error, expanding the fitting range')
            converged_parameters_linear_bl = startingpoint
            derspec.working_range = (derspec.working_range[0]-interpoint_distance, derspec.working_range[1]+interpoint_distance)
            derpeak.wn = derspec.x
            continue

    #@Test&Debug # Optional plot_da_peak
    if display > 1: 
        the_baseline = (derpeak.BL_offset + derpeak.BL_slope*(derspec.x - derpeak.position))
        plt.plot(derspec.x, derspec.y, 'ko',
                  derspec.x, derpeak.curve_with_BL, 'r:', 
                  derspec.x, the_baseline, 'k-',  mfc='none')
        # plt.xlabel('Raman shift / cm$^{-1}$')
        # plt.ylabel('intensity')
        plt.title('fit single peak')
        plt.show()
    if display > 0: 
        print(' position: ', derpeak.position)
        print(' fwhm: ', derpeak.fwhm)
        print(' peak area: ', derpeak.peak_area)
        if display > 1:
            print(' peak asymmetry: ', derpeak.asymmetry)
            print(' peak Gaussian share: ', derpeak.Gaussian_share)
            print(' baseline slope: ', derpeak.BL_slope)
            print(' baseline offset: ', derpeak.BL_offset)
            print(' peak height: ', derpeak.peak_height)
            print(' Voigt_amplitute: ', derpeak.voigt_amplitude)
        
    # restoring the working range of the ExpSpec class:
    derspec.working_range = original_working_range
    return derpeak


if __name__ == '__main__':
    
    s2test = np.genfromtxt('graphene_2x240s.txt')
    wn = s2test[:,0]
    graphene_w_cosmics = s2test[:,1]

    from expspec import ExpSpec
    
    graphene_with_cosmics = ExpSpec(wn, graphene_w_cosmics)

    derpeak = fit_single_peak(graphene_with_cosmics, peak_position=1600, fwhm=20, fitrange=(1400, 1700), display=2)
    print('...\n This function finds parameters of a single peak in the spectrum')
    print('...\n It returns these parameters defined in the class CalcPeak \n   and optionally prints them to the console')
    
                
    
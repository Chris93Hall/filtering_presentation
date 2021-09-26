"""
plot.py
"""
from __future__ import absolute_import

import comms
import matplotlib.pyplot as plt
import numpy as np

def abs_re_im(signal, **kwargs):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.abs(signal))
    plt.subplot(3, 1, 2)
    plt.plot(np.real(signal))
    plt.subplot(3, 1, 3)
    plt.plot(np.imag(signal))
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def mag_phase(signal, xaxis=None, **kwargs):
    plt.figure()
    plt.subplot(2, 1, 1)
    if xaxis is not None:
        plt.plot(xaxis, np.abs(signal))
    else:
        plt.plot(np.abs(signal))
    plt.subplot(2, 1, 2)
    if xaxis is not None:
        plt.plot(xaxis, np.angle(signal))
    else:
        plt.plot(np.angle(signal))
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def fft(signal, **kwargs):
    """
    special kwargs:
        num_points - number of dft samples to compute
    """
    if 'num_points' in kwargs.keys():
        num_points = kwargs['num_points']
    else:
        num_points = None
    signal_freq, freq_axis = comms.dft.fft(signal, bins=num_points, get_axis=True)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freq_axis, np.abs(signal_freq))
    plt.subplot(2, 1, 2)
    plt.plot(freq_axis, np.angle(signal_freq))
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def matrix_mag(mat, **kwargs):
    plt.figure()
    plt.imshow(np.abs(mat), interpolation='none')
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def matrix_power(mat, **kwargs):
    plt.figure()
    plt.imshow(mat*mat, interpolation='none')
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def stem(signal, **kwargs):
    plt.figure()
    plt.stem(range(len(signal)), signal)
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def constellation(vector, **kwargs):
    plt.figure()
    plt.plot(np.real(vector), np.imag(vector), 'o')
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def plot_beampattern(powervector, anglevector, style="polar", db_range=20):
    """
    Inputs:
        - powervector      vector in linear scale
        - anglevector      angles in degrees
    TODO:
        - axis label starting at 0
    """
    powervector = np.array(powervector)
    powervector = 10*np.log10(powervector)
    powervector = powervector - np.max(powervector)
    powervector = powervector + db_range
    powervector[powervector < 0] = 0.0
    plt.figure()
    if style == "polar":
        anglevector = anglevector*2*np.pi/360
        plt.subplot(111, projection='polar')

    if len(powervector.shape) == 1:
        plt.plot(anglevector, powervector, '-')
    else: # multiple beampatterns to plot
        for index in range(powervector.shape[0]):
            plt.plot(anglevector, powervector[index, :], '-')
    plt.show(block=False)
    return

def beampattern_2D(power_array, style='flat'):
    """
    TODO: add contour plot option
    """
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    power_array = np.array(power_array)
    power_array = 10*np.log10(power_array)
    power_array = power_array + 20
    power_array[power_array < 0] = 0.0

    if style == 'flat':
        plt.figure()
        plt.imshow(power_array)
        plt.show(block=False)
    else:
        array_shape = np.shape(power_array)        
        x, y = np.meshgrid(np.arange(array_shape[0]), np.arange(array_shape[1]))
        z = power_array

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)
        plt.show(block=False)
    return

def ambiguity_function(vector, db_scale=False, **kwargs):
    ambiguity = comms.dsp.calc_ambiguity_function(vector, db_scale=db_scale)
    #ambiguity[ambiguity<np.max(ambiguity)-50] = np.max(ambiguity)-50
    shape = np.shape(ambiguity)
    plt.figure()
    plt.imshow(ambiguity, cmap='gray')
    plt.xticks([0, shape[0]], [-shape[0]/2.0, shape[0]/2.0])
    plt.yticks([0, shape[1]], ['-3.14', '3.14'])
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def cross_ambiguity_function(vector1, vector2, **kwargs):
    ambiguity = comms.dsp.calc_cross_ambiguity_function(vector1, vector2, db_scale=False)
    #ambiguity[ambiguity<np.max(ambiguity)-50] = np.max(ambiguity)-50
    plt.figure()
    plt.imshow(ambiguity, cmap='gray')
    kwarg_parser(kwargs)
    plt.show(block=False)
    return

def ambiguity_function2(vector, **kwargs):
    """
    Testing matplotlib surf plot
    TODO
    https://matplotlib.org/examples/mplot3d/surface3d_demo.html 
    """
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    ambiguity = comms.dsp.calc_ambiguity_function(vector)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, len(vector))
    Y = np.arange(0, len(vector))
    X, Y = np.meshgrid(X, Y)
    return

def normal_distributions(mean_var_pairs, **kwargs):
    means_list, vars_list = zip(*mean_var_pairs)
    x_min = np.min(means_list) - (np.max(means_list) - np.min(means_list))*1.5
    x_max = np.max(means_list) + (np.max(means_list) - np.min(means_list))*1.5
    x_step_size = np.abs(x_max - x_min)/1000
    x_axis = np.arange(x_min, x_max, x_step_size)
    plt.figure()
    for mean_var in mean_var_pairs:
        mean = mean_var[0]
        var = mean_var[1]
        y = (1./np.sqrt(2*np.pi*var))*np.e**(-((x_axis-mean)**2)/(2*var))
        plt.plot(x_axis, y)
    plt.show(block=False)
    return

def prob_mass_func(samples):
    """
    Given a list of samples from a probability mass function, plot the
    probability mass function as a stem plot.
    """
    key = list(set(list(samples)))
    key.sort()
    counts = [0]*len(key)
    for val in samples:
        index = key.index(val)
        counts[index] += 1

    plt.figure()
    plt.stem(key, counts)
    plt.show(block=False)
    return

def kwarg_parser(kwarg_dict):
    if 'title' in kwarg_dict:
        plt.title(kwarg_dict['title'])
    if 'xlabel' in kwarg_dict:
        plt.xlabel(kwarg_dict['xlabel'])
    if 'ylabel' in kwarg_dict:
        plt.ylabel(kwarg_dict['ylabel'])
    if 'grid' in kwarg_dict:
        plt.grid(kwarg_dict['grid'])
    return

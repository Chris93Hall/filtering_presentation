"""
lpf_demo1.py
"""

import numpy as np
import scipy as sp
import scipy.signal

import plot
import signal_generator

def build_raised_cosine(length, upsample_factor, rolloff_factor):
    sample_period = 1.0/upsample_factor
    x_axis = np.arange(0, length*sample_period, sample_period) - length*sample_period/2.0
    raised_cos_vector = [0]*len(x_axis)
    for index in range(len(x_axis)):
        x = x_axis[index]
        if rolloff_factor != 0.0 and abs(x) == 1.0/(2.0*rolloff_factor):
            positive_check = 1.0/(2.0*rolloff_factor)
            negative_check = -positive_check
            if positive_check == x:
                raised_cosine = (np.pi/4.0)*np.sin(np.pi/(2*rolloff_factor))/(np.pi/(2*rolloff_factor))
            if negative_check == x:
                raised_cosine = (np.pi/4.0)*np.sin(np.pi/(2*rolloff_factor))/(np.pi/(2*rolloff_factor))
        elif 0.0 == x:
            raised_cosine = 1.0
        else:
            raised_cosine = np.sin(np.pi*x)/(np.pi*x)
            raised_cosine = raised_cosine*np.cos(np.pi*rolloff_factor*x)
            raised_cosine = raised_cosine/(1-((2.0*rolloff_factor*x)**2))
        raised_cos_vector[index] = raised_cosine 
    return raised_cos_vector

def lpf_demo1():
    filt = build_raised_cosine(32, upsample_factor=4, rolloff_factor=0.0)
    sig = signal_generator.sinusoid(128, 0.4*np.pi)
    plot.stem(filt, title='Moving Average Filter With 5 Taps')
    plot.stem(sig, title='Input Signal')
    output = np.convolve(filt, sig, mode='full') # mode can be 'full', 'same', 'valid'
    plot.stem(output, title='Output Signal')
    ww, hh = scipy.signal.freqz(filt)
    plot.mag_phase(hh, title='Magnitude and Phase of Transfer Function', xaxis=ww/np.pi)
    a = input()
    return

if __name__ == '__main__':
    lpf_demo1()

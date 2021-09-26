"""
bpf_demo.py
"""

import numpy as np
import scipy as sp
import scipy.signal

import plot
import signal_generator
from window_generator import WindowGenerator

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

def bpf_demo1():
    filt = build_raised_cosine(32, upsample_factor=4, rolloff_factor=0.0)
    hann_win = WindowGenerator.hann(32)
    filt = filt*hann_win

    ww, hh = scipy.signal.freqz(filt)
    plot.stem(filt, title='Sinc Filter With Hanning Window')
    plot.mag_phase(hh, xaxis=ww/np.pi)

    filt = filt*np.exp(1j*0.4*np.pi*np.arange(32))
    ww, hh = scipy.signal.freqz(filt)
    plot.stem(filt, title='Sinc Filter With Hanning Window')
    plot.mag_phase(hh, xaxis=ww/np.pi)

    sig = signal_generator.sinusoid(128, 0.4*np.pi)
    output1 = np.convolve(filt, sig, mode='full') # mode can be 'full', 'same', 'valid'
    plot.stem(output1, title='Output Signal')
    a = input()
    return

if __name__ == '__main__':
    bpf_demo1()

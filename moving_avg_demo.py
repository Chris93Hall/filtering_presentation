"""
moving_avg_demo.py
"""

import numpy as np
import scipy as sp
import scipy.signal

import plot
import signal_generator

def moving_average_builder(length):
    filt = np.array([1.0/length]*length)
    return filt

def moving_average_demo1():
    filt = moving_average_builder(5)
    sig = signal_generator.sinusoid(128, 0.4*np.pi)
    plot.stem(filt, title='Moving Average Filter With 5 Taps')
    plot.stem(sig, title='Input Signal')
    output = np.convolve(filt, sig, mode='full') # mode can be 'full', 'same', 'valid'
    plot.stem(output, title='Output Signal')
    ww, hh = scipy.signal.freqz(filt)
    plot.mag_phase(hh, xaxis=ww/np.pi)
    a = input()
    return

if __name__ == '__main__':
    moving_average_demo1()

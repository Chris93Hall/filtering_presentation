"""
matched_filt_demo.py
"""

import random
import numpy as np
import scipy as sp
import scipy.signal

import plot
import signal_generator

def generate_signal():
    signal = np.zeros(1024)
    template = np.array([0.5, 1.0, 1.0, -0.5, 1.0, 1.0, -1.0, 1.0, -1.0, -0.5, -0.25])
    for _ in range(10):
        index = round(random.uniform(0,1024-12))
        signal[index:index+11] = template
    return signal

def matched_filt_demo1():
    filt = np.array([-0.25, -0.5, -1.0, 1.0, -1.0, 1.0, 1.0, -0.5, 1.0, 1.0, 0.5])
    plot.stem(filt, title='Matched Filter')
    ww, hh = scipy.signal.freqz(filt)
    plot.mag_phase(hh, xaxis=ww/np.pi, title='Filter Frequency Response')

    sig = generate_signal()
    plot.stem(sig, title='Input Signal')
    output1 = np.convolve(filt, sig, mode='full') # mode can be 'full', 'same', 'valid'
    plot.stem(output1, title='Output Signal')

    sig = sig + signal_generator.gaussian_noise(1024, variance=0.25)
    plot.stem(sig, title='Input Signal Plus Noise')
    output2 = np.convolve(filt, sig, mode='full') # mode can be 'full', 'same', 'valid'
    plot.stem(output2, title='Output Signal')
    a = input()
    return

if __name__ == '__main__':
    matched_filt_demo1()

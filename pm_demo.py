"""
pm_demo.py
"""

import numpy as np
import scipy as sp
import scipy.signal

import plot
import signal_generator
from window_generator import WindowGenerator

def pm_demo1():
    num_taps = 32
    bands = [0.0, 0.4, 0.6, 1.0]
    #bands = [0.0, 0.4, 0.45, 1.0]
    desired = [1.0, 0.0]
    weight = [1.0, 1.0]
    #weight = [1.0, 2.0]
    hz = 2
    filt = scipy.signal.remez(num_taps, bands, desired, weight=weight, Hz=hz)

    ww, hh = scipy.signal.freqz(filt)
    plot.stem(filt, title='Sinc Filter With Hanning Window')
    plot.mag_phase(hh, xaxis=ww/np.pi)
    a = input()
    return

if __name__ == '__main__':
    pm_demo1()

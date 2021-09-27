"""
window_demo.py
"""

import numpy as np
import scipy as sp
import scipy.signal

import plot
import signal_generator
from window_generator import WindowGenerator

def window_demo1():
    length = 32

    rect_win = WindowGenerator.rectangle(length)
    ww, rect_win_f = scipy.signal.freqz(rect_win)
    plot.stem(rect_win, title='Rectangular Window')
    plot.mag_phase(rect_win_f, xaxis=ww/np.pi)

    tri_win = WindowGenerator.triangle(length)
    ww, tri_win_f = scipy.signal.freqz(tri_win)
    plot.stem(tri_win, title='Triangle Window')
    plot.mag_phase(tri_win_f, xaxis=ww/np.pi)

    hann_win = WindowGenerator.hann(length)
    ww, hann_win_f = scipy.signal.freqz(hann_win)
    plot.stem(hann_win, title='Hanning Window')
    plot.mag_phase(hann_win_f, xaxis=ww/np.pi)

    parzen_win = WindowGenerator.parzen(length)
    ww, parzen_win_f = scipy.signal.freqz(parzen_win)
    plot.stem(parzen_win, title='Parzen Window')
    plot.mag_phase(parzen_win_f, xaxis=ww/np.pi)

  

    a = input()
    return

if __name__ == '__main__':
    window_demo1()

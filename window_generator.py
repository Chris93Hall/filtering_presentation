"""
Contains functions for generating windowing functions
"""

import numpy as np

class WindowGenerator():
        
    @staticmethod
    def rectangle(length):
        signal = np.ones(length)
        return signal
                
    @staticmethod
    def triangle(length):
        index = np.array(range(length))
        signal = 1 - np.abs((index - (0.5*(length-1)))/(0.5*(length-1)))
        return signal
                
    @staticmethod
    def parzen(length):
        signal = np.zeros(length)
        half_length = float(length/2)
        for index in range(-length//2, length//2):
            if np.abs(index) <= length/4:
                signal[index+(length//2)] = 1-6*((np.abs(index)/half_length)**2)*(1-(np.abs(index)/half_length))
            else:
                signal[index+(length//2)] = 2*(1-(np.abs(index)/half_length))**3
        return signal
        
    @staticmethod
    def welch(length):
        index = np.array(range(length))
        half_length = float((length-1)/2)
        signal = 1 - ((index-half_length)/half_length)**2
        return signal
                
    @staticmethod
    def sine(length):
        index = np.array(range(length))
        signal = np.sin(np.pi*index/float(length-1))
        return signal
                
    @staticmethod
    def power_of_sine(length, power):
        index = np.array(range(length))
        signal = np.sin(np.pi*index/float(length-1))**power
        return signal
        
    @staticmethod
    def hann(length):
        index = np.array(range(length))
        signal = 0.5-(1-0.5)*np.cos(2*np.pi*index/(length-1))
        return signal

    @staticmethod
    def taylor():
        """
        TODO
        """
        return


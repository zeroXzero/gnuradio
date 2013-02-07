from gnuradio import gr
import numpy as np
import scipy as sp
import scipy.special as scsp
import scipy.signal as scsig

class dopfirdes():
    '''
        Doppler filter class
    '''
    def __init__(self,max_doppler, order):
        self.max_dop = max_doppler
        self.ntaps = order/2
        self.fcutoff = {"flat": self.max_dop, "jakes": self.max_dop,
                        "gauss": self.max_dop * (np.log(2))**0.5}

    def jakes(self):
        samp_rate = 10 * self.fcutoff["jakes"]
        samp_period = 1.0/samp_rate
        pos = scsp.gamma(0.75)*(self.max_dop**0.25)*np.array(
                  [scsp.jn(0.25, 2 * np.pi * self.max_dop * i * samp_period)/
                  ((np.pi*i*samp_period)**0.25) for i in range(1,self.ntaps+1)])
        pos0 = scsp.gamma(0.75)/scsp.gamma(5.0/4) * (self.max_dop**0.5)
        self.h = np.append(np.append(pos[::-1],pos0),pos) * sp.hamming(self.ntaps*2+1)
        #normalize taps
        self.h = self.h/np.power(np.sum(np.power(np.abs(self.h),2)),0.5)
        return (self.h, samp_rate)

    def gaussian(self, sigma = None):
        if sigma == None:
            sigma = self.max_dop/np.sqrt(2)
            samp_rate = 10 * self.fcutoff["gauss"]
            samp_period= 1.0/samp_rate
        else:
            samp_rate = 10 * sigma * (2*np.log(2))**0.5 
            samp_period= 1.0/samp_rate
        pos = (2*np.pi)**0.25 * np.sqrt(sigma) * np.array(
                  [ np.sinc(-4 * np.power(np.pi,2) * np.power(sigma,2) * 
                    np.power(i*samp_period,2)) for i in range(0,self.ntaps+1)])
        self.h = np.append(pos[:0:-1],pos) * sp.hamming(self.ntaps*2+1)
        #normalize taps
        self.h = self.h/np.power(np.sum(np.power(np.abs(self.h),2)),0.5)
        return (self.h, samp_rate)

    def flat(self):
        samp_rate = 10 * self.fcutoff["flat"]
        samp_period = 1.0/samp_rate
        pos = (2*self.max_dop)**0.5 * np.array(
                  [ np.sinc(2*self.max_dop*samp_period*i) for i in range(0,self.ntaps+1)])
        self.h = np.append(pos[:0:-1],pos) * sp.hamming(self.ntaps*2+1)
        #normalize taps
        self.h = self.h/np.power(np.sum(np.power(np.abs(self.h),2)),0.5)
        return (self.h, samp_rate)

import numpy as np
import scipy as sp
import scipy.special as scsp
import scipy.signal as scsig
from gnuradio import gr
from gnuradio import gr 
from gnuradio import blocks 
from gnuradio import analog 
from gnuradio import filter 
from gnuradio.channels import doppler_firdes 
from gnuradio.channels import fading_tdl 

class rician_model(gr.hier_block2):
    def __init__(self, 
            samp_rate = None, #sample rate of the input signal
            kfactor = None,   #ratio of power in los and diffuse
            los_dop = None,   #doppler shifts of los path
            los_initphase = [0], # initial phase of los path
            max_dop = None,   #maximum diffuse doppler shift
            pdelay = None,    #path delay vector
            pgain_db = None,  #path gain in db
            seed = 100,      #seed for simulation
            ntaps = 7,     #odd tdl tap length
            dspectrum = None,  #default spectrum is jakes for all paths
            dspec_out = False  #set output for path coefficients
            ):
        gr.hier_block2.__init__(self, "Rayleigh_model",
                       gr.io_signature(1, 1, gr.sizeof_gr_complex),
                       gr.io_signature(len(pdelay)*int(dspec_out)+1 if pdelay else 1, 
                                       len(pdelay)*int(dspec_out)+1 if pdelay else 1, 
                                       gr.sizeof_gr_complex))

        self.samp_rate = samp_rate
        self.ntaps = ntaps
        self.kfactor = kfactor
        self.los_dop = los_dop 
        self.los_initphase = los_initphase 
        self.max_dop = max_dop
        self.pdelay = pdelay
        self.pgain_db = pgain_db
        self.seed = seed
        self.order = 64
        self.max_poly_interp = 20
        self.min_poly_interp = 10

        self.sanitize_inputs(pgain_db, pdelay, dspectrum)

        #generating alpha_kn as per reference
        self.alpha_kn = []
        self.g_n  = []
        for n in range(-(self.ntaps-1)/2,((self.ntaps-1)/2)+1):
            alpha_mul = [blocks.multiply_const_cc(np.sinc(self.samp_rate * self.pdelay[k] - n)) for k in range(self.npaths)]
            self.alpha_kn.append(alpha_mul)
            self.g_n.append(blocks.add_cc())
        
        self.dspec = doppler_firdes.dopfirdes(max_dop, self.order)
        self.fn = {"flat":self.dspec.flat, "jakes":self.dspec.jakes,
                         "default":self.dspec.jakes}

        #doppler sample rate based on cutoff frequency
        self.fds = np.array([ self.tapcall(self.pathspec[i])[1] 
                     for i in range(self.npaths)])

        #interpolation factor for each path
        self.interp_factor = np.floor(self.samp_rate/self.fds)

        self.lin_interp = np.array([])
        self.poly_interp = np.array([])
        for i in range(self.npaths):
            if self.interp_factor[i] <= self.max_poly_interp:
                self.lin_interp = np.append(self.lin_interp,1) 
                self.poly_interp = np.append(self.poly_interp,self.interp_factor[i])
            else:
                self.lin_interp = np.append(self.lin_interp,np.round(self.interp_factor[i]/self.min_poly_interp)) 
                self.poly_interp = np.append(self.poly_interp,self.min_poly_interp)

        #recalculating doppler sample rate
        self.fds = self.samp_rate/(self.lin_interp * self.poly_interp)

        #independent gaussian source (check seed)
        self.rands = [analog.noise_source_c(analog.GR_GAUSSIAN, 1,self.seed*i*(-1)**i) 
                      for i in range(self.npaths)]

        #throttle to reduce sample rate 
#self.thr = [gr.throttle(gr.sizeof_gr_complex, self.fds[i])
#for i in range(self.npaths)]

        self.dsfilt= [ filter.fir_filter_ccc(1,self.tapcall(self.pathspec[i])[0]) 
                       for i in range(self.npaths)] 

        self.gainmult = []
        self.losadd = []
        for i in range(self.npaths):
            self.gainmult.append(blocks.multiply_const_cc(np.sqrt(
                                                     np.power(10,self.pgain_db[i]/20.0) / 
                                                     (self.kfactor[i]+1)
                                                     )))
            self.losadd.append(blocks.add_const_cc(np.sqrt(
                                            np.power(10,self.pgain_db[i]/20.0) * self.kfactor[i]/
                                            (self.kfactor[i]+1)) * 
                                            np.exp((2*np.pi*self.los_dop[i] + self.los_initphase[i])*1j)))


        #Fix to push zeros to tdl channel memory (history) with zeros
        self.inszeros= gr.delay(gr.sizeof_gr_complex,self.ntaps-1) 
        self.tdl = fading_tdl.tdl_channel(self.ntaps)

        self.polyf = []
        self.linf = []
        for i in range(self.npaths):
            polytaps = filter.firdes.low_pass_2(self.poly_interp[i], 
                                            self.poly_interp[i] * self.fds[i],
                                            self.fds[i]/2.0, self.fds[i]/10,attenuation_dB=80,
                                            window=filter.firdes.WIN_BLACKMAN_hARRIS)
            lintaps = [1 - float(j)/self.lin_interp[i] for j in np.arange(self.lin_interp[i])]
            self.polyf.append(filter.pfb_interpolator_ccf(int(self.poly_interp[i]), polytaps))
            self.linf.append(filter.interp_fir_filter_ccf(int(self.lin_interp[i]), lintaps))

        self.add = blocks.add_cc()

        #connect all blocks
        self.connect(self,self.inszeros,(self.tdl,0))
        for i in range(self.npaths):
#self.connect(self.rands[i],self.thr[i],self.dsfilt[i],self.polyf[i],
            self.connect(self.rands[i],self.dsfilt[i],self.polyf[i],
                    self.linf[i],self.gainmult[i], self.losadd[i])
            if dspec_out:
                self.connect(self.losadd[i],(self,i+1))

        for n in range(self.ntaps):
            for k in range(self.npaths):
                self.connect(self.losadd[k],(self.g_n[n],k))
            self.connect(self.g_n[n],(self.tdl,n+1))
            self.connect((self.tdl,n),(self.add,n))

        self.connect(self.add,(self,0))

    def set_kfactor(self, kfactor):
        self.kfactor = kfactor

    def set_los_dop(self, los_dop):
        self.los_dop = los_dop

    def set_los_initphase(self, los_initphase):
        self.los_initphase = los_initphase

    def set_max_dop(self, max_dop):
        self.max_dop = max_dop

    def set_pdelay(self, pdelay):
        self.pdelay  = pdelay

    def set_pgain_db(self, pgain_db):
        self.pgain_db = pgain_db

    def set_dspectrum(self, dspectrum):
        self.dspectrum = dspectrum

    def model(self):
        return self.model

    def samp_rate(self):
        return self.samp_rate
    
    def kfactor(self):
        return self.kfactor

    def los_dop(self):
        return self.los_dop

    def los_initphase(self):
        return self.los_initphase

    def max_dop(self):
        return self.max_dop

    def pdelay(self):
        return self.pdelay

    def pgain_db(self):
        return self.pgain_db

    def dspectrum(self):
        return self.dspectrum

    def sanitize_inputs(self, pgain_db, pdelay, dspectrum):
        assert len(pgain_db) == len(pdelay), 'Length different for delay and gain'
        self.npaths = len(pdelay)

        if dspectrum:
            self.pathspec = dspectrum.strip("[]").replace(" ", "").split(",")
        else:
            self.pathspec = ["default"]*self.npaths

        #if specified path doppler is less repeat the last specified one
        # for all non-specified paths
        if len(self.pathspec) != self.npaths:
            self.pathspec = self.pathspec + [self.pathspec[-1]]*(self.npaths - len(self.pathspec))

        #make kfactor and other los parameter zero to use a generic gain multiplier
        if len(self.kfactor) != self.npaths:
            self.kfactor = self.kfactor + [0]*(self.npaths - len(self.kfactor))

        if len(self.los_dop) != self.npaths:
            self.los_dop = self.los_dop + [0]*(self.npaths - len(self.los_dop))

        if len(self.los_initphase) != self.npaths:
            self.los_initphase = self.los_initphase + [0]*(self.npaths - len(self.los_initphase))

    def tapcall(self, dftype):

        dftype = dftype.replace('(',' ').replace(')', ' ').split()
        if dftype[0] == "gauss":
            if len(dftype) > 1:
                return self.dspec.gaussian(float(dftype[1]))
            else:
                return self.dspec.gaussian()
        else:
            return self.fn[dftype[0]]()

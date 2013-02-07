import numpy as np
import scipy as sp
import scipy.special as scsp
import scipy.signal as scsig
from gnuradio import gr

class tdl_channel(gr.basic_block):
    def __init__(self, ntaps):
        gr.basic_block.__init__(self, name="tdl_channel", 
                in_sig=[np.complex64]+[np.complex64]*ntaps, 
                out_sig=[np.complex64]*ntaps)
        self.ntaps = ntaps 
#self.set_auto_consume(False)

    def forecast(self, noutput_items, ninput_items_required):
        ninput_items_required[0] = noutput_items + self.ntaps -1
        for i in range(1,len(ninput_items_required)):
            ninput_items_required[i] = noutput_items

    def general_work(self, input_items, output_items):

        #channel will not push zeros  (self.ntaps-1) zeros
        #done outside the tdl channel
        for i in range(len(output_items[0])):
            taps = np.array([ input_items[j][i] for j in range(1,self.ntaps+1)])
            out = np.multiply(input_items[0][i:i+self.ntaps],taps)
            for j in range(self.ntaps):
                output_items[j][i] = out[j]
#print "taps",taps
#print "input",input_items[0][i:i+self.ntaps]
#print "mult", np.multiply(input_items[0][i:i+self.ntaps],taps)
        self.consume_each(len(output_items[0]))

        return len(output_items[0])

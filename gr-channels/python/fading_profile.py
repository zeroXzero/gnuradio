from gnuradio.channels import rician_model 
from gnuradio.channels import rayleigh_model 

def COST207_RAx4(samp_rate, max_dop, ntaps, dspec_out):
    model = rician_model.rician_model(samp_rate = samp_rate,
                            kfactor = ([0.87/0.13]),
                            los_dop = ([0.7 * max_dop]),
                            max_dop = max_dop,
                            pdelay  = ([0.0, 0.2e-6, 0.4e-6, 0.6e-6]),
                            pgain_db = ([0, -2, -10, -20]),
                            ntaps = ntaps,
                            dspec_out = dspec_out)
    return model 

def COST207_RAx6(samp_rate, max_dop, ntaps, dspec_out):
    model = rician_model.rician_model(samp_rate = samp_rate,
                         kfactor = ([0.87/0.13]),
                         los_dop  = ([0.7 * max_dop]),
                         max_dop = max_dop,
                         pdelay  = ([0.0, 0.1e-6, 0.2e-6, 0.3e-6, 0.4e-6, 0.5e-6]),
                         pgain_db = ([0, -4, -8, -12, -16, -20]),
                         ntaps = ntaps,
                         dspec_out = dspec_out)
    return model 


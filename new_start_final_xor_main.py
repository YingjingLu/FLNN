from data_source.spiral_data_source import * 
import numpy as np 
import os as os
from model.new_start_final_xor import * 
from model.baseline_xor import *
from opts.new_start_final_xor_opts import *
from opts.baseline_xor_opts import *
from image_reconstructor import *

BASELINE = 1

if __name__ == "__main__":

    if BASELINE:
        opts = Baseline_Xor_Opts()
    else:
        opts = New_Start_Final_Xor_Opts() 
    opts.data_source = Data_Source( opts )
    opts.data_source.load_unsplit_samples()
    if BASELINE:
        f = Baseline_Xor( opts )
    else:
        f = New_Start_Final_Xor( opts )

    opts.train = 0
    if opts.train:
        f.train()
    else:
        f.saver.restore( f.sess, save_path = "cpt/baseline_xor_larger/380000/model.ckpt" )
        w0 = f.sess.run( f.w_l0 )
        w1 = f.sess.run( f.w_l1 )
        np.save( "w0.npy", w0 )
        np.save( "w1.npy", w1 )

        a = np.load("w0.npy")
        get_heat_map( a, "xor_w0.png", [ 16, 3] )

        b = np.load("w1.npy")
        get_heat_map( b, "xor_w1.png", [ 8, 16 ] )

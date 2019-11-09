from data_source.baseline_mnist_dataSource import * 
import numpy as np 
import os as os
from model.new_start_final_cifar_10 import * 
from model.cifar_10_baseline import *
from model.senn_cifar import *
from opts.new_start_final_cifar_10_opts import *
from opts.cifar_10_baseline_opts import *
from image_reconstructor import *
GATE_ANALYSIS = 1
BASELINE = 0
SENN = 0

if __name__ == "__main__":
    if BASELINE:
        opts = Cifar_10_Baseline_Opts()
    else:
        opts = New_Start_Final_Cifar_10_Opts() 
    if SENN:
        opts.cpt_path = "cpt/senn_cifar"
    opts.data_source = Data_Source( opts )
    opts.data_source.load_unsplit_samples()
    if BASELINE:
        f = Cifar_10_Baseline( opts )
    elif SENN:
        f = SENN_CIFAR( opts ) 
    else:
        f  = New_Start_Final_Cifar_10( opts )
    opts.train = 0
    if opts.train:
        f.train()
    else:
        f.saver.restore( f.sess, save_path = "cpt/new_start_final_cifar_10/800000/model.ckpt" )
        in_sample, in_label = f.opts.data_source.get_test()
        # f.get_predict( in_sample, in_label )
        l0_feat_gate, l1_feat_gate, final_w = f.sess.run( [ f.l0_obj.feature_mask_test, f.l1_obj.feature_mask_test, f.final_w ],
                                                          feed_dict = { f.in_sample : in_sample, f.in_label: in_label } )
        l0_feat_gate = l0_feat_gate.reshape( -1, 1 )
        l1_feat_gate = l1_feat_gate.reshape( -1, 1 )

        l0_w = final_w[:3072, :]
        l1_w = final_w[3072:5120, :]
        l2_w = final_w[ 5120:, : ]

        l0_w = l0_w * l0_feat_gate
        l1_w = l1_w * l1_feat_gate
        print(l0_w.shape)
        print( np.mean( np.abs( l0_w ) ) )
        print( np.mean( np.abs( l1_w ) ) )
        print( np.mean( np.abs( l2_w ) ) )

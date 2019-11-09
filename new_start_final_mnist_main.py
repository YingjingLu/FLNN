from data_source.baseline_mnist_dataSource import * 
import numpy as np 
import os as os
from model.new_start_l0_mnist import * 
from model.new_start_final_mnist import * 
from opts.new_start_l0_opts import *
from model.senn_mnist import *
from image_reconstructor import *
GATE_ANALYSIS = 0
SENN = 0


if __name__ == "__main__":

    opts = New_Start_L0_Opts() 
    opts.data_source = Data_Source( opts )
    opts.data_source.load_unsplit_samples()
    if SENN:
        opts.cpt_path = "cpt/senn_mnist"
        f = SENN_MNIST( opts )
    else:
        f = New_Start_Final_Mnist( opts )
    if GATE_ANALYSIS == 1:
        opts.train = 0
    if opts.train:
        f.train()
    else:
        f.saver.restore( f.sess, save_path = "cpt/new_start_final_mnist_2/400000/model.ckpt" )


        in_sample, in_label = f.opts.data_source.get_test()
        accu = f.predict( in_sample, in_label )
        print( "Accu test", accu )
        l0_feat_gate, l1_feat_gate, final_w = f.sess.run( [ f.l0_obj.feature_mask_test, f.l1_obj.feature_mask_test, f.final_w ],
                                                          feed_dict = { f.in_sample : in_sample, f.in_label: in_label } )
        l0_feat_gate = l0_feat_gate.reshape( -1, 1 )
        l1_feat_gate = l1_feat_gate.reshape( -1, 1 )

        l0_w = final_w[:784, :]
        l1_w = final_w[784:1084, :]
        l2_w = final_w[ 1084:, : ]

        l0_w = l0_w * l0_feat_gate
        l1_w = l1_w * l1_feat_gate
        print(l0_w.shape)
        print( np.mean( np.abs( l0_w ) ) )
        print( np.mean( np.abs( l1_w ) ) )
        print( np.mean( np.abs( l2_w ) ) )

        # get_gate_map( l0_feat_gate.reshape( 28, 28 ), "mnist/l0_net_gate_map.png" )
        # exit()
        # # analyze l0 feature:
        # for i in range(10):
        #     l0_weight = final_w[ :784, i ]
            
        #     weight_pic = l0_weight * l0_feat_gate 
        #     get_heat_map( weight_pic.reshape( 28, 28 ), "mnist/l0_gated_weight_" + str(i) + "_map.png" )

        

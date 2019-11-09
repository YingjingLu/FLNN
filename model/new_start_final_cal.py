import tensorflow as tf 
import numpy as np 

import tensorflow as tf 
import numpy as np 
import os
from model.new_start_models import *

class New_Start_Final_Cal( object ):
    
    def __init__(self, opts):
        self.sess = tf.Session() 
        self.opts = opts 
        self.init()
        self.saver = tf.train.Saver( max_to_keep = 100 )

    
    def init( self ):
        self.add_input_placeholder()
        self.construct_clf()
        self.construct_loss()
        self.sess.run( [ tf.global_variables_initializer() ] )
        os.mkdir( self.opts.cpt_path ) if not os.path.exists( self.opts.cpt_path ) else print()
    
    def add_input_placeholder( self ):
        opts= self.opts
        self.in_sample = tf.placeholder( tf.float32, [ None ] + opts.sample_shape )
        self.in_label = tf.placeholder( tf.float32, [ None ] + opts.label_shape )

    def construct_clf( self ):
        
        with tf.variable_scope( "nn" ) as scope:
            self.l0_obj = Group_L0( 13, 64, activation = tf.nn.relu, weight_decay = 1.0 , name = "l0" )
            self.net_l0_train, self.feat_l0_train, self.net_l0_test, self.feat_l0_test, self.l0_regu = self.l0_obj.build( self.in_sample, self.in_sample )
            
            self.l1_obj = Group_L0( 64, 32, activation = tf.nn.relu, weight_decay = 1.0, name = "l1" )
            self.net_l1_train, self.feat_l1_train, self.net_l1_test, self.feat_l1_test, self.l1_regu = self.l1_obj.build( self.net_l0_train, self.net_l0_test )

            self.concat_train = tf.concat( [ self.feat_l0_train, self.feat_l1_train, self.net_l1_train ], axis = 1 ) 
            self.concat_test = tf.concat( [ self.feat_l0_test, self.feat_l1_test, self.net_l1_test ], axis = 1 )
            # self.logit_obj = Group_L0( 784 + 300 + 100, 10, weight_decay = 5e-4, name = "logit" )
            self.logit_train, _, self.final_w, _ =  dense( self.concat_train, 1, initializer = tf.random_normal_initializer( stddev = 1./tf.sqrt( 1./2. ) ), name = "logit" )
            self.logit_test, _, _, _ =  dense( self.concat_test, 1, initializer = tf.random_normal_initializer( stddev = 1./tf.sqrt( 1./2. ) ), name = "logit", reuse = True )
            self.prediction = self.logit_test
    
    def construct_loss( self ):
        self.classifier_loss = tf.reduce_mean( tf.square( self.logit_train - self.in_label ) )
        self.classifier_loss = self.classifier_loss \
                        -0.5*( tf.reduce_mean( self.l0_regu ) + tf.reduce_mean( self.l1_regu ) )
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim_nn = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.classifier_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') )

    def calc_sparsity( self, in_sample, in_label, thresh = 0.98 ):
        l0_feat_gate_test, l1_feat_gate_test = self.sess.run( [ self.l0_obj.feature_mask_test, self.l1_obj.feature_mask_test ], 
                                                              feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
        l0_feat_gate_test = l0_feat_gate_test.ravel()
        l1_feat_gate_test = l1_feat_gate_test.ravel()
        print( l0_feat_gate_test[ :10 ] )
        l0 = np.sum( l0_feat_gate_test >= thresh )
        l1 = np.sum( l1_feat_gate_test >= thresh )

        print( "l0 gate", l0, "out of", l0_feat_gate_test.shape[0], "l1_gate", l1, "out of", l1_feat_gate_test.shape )
        self.feat_gating_list.append( [ l0, l1 ] )

    def train( self ):
        self.loss_list = []
        self.mse_list = []
        self.feat_gating_list = []
        min_mse = 1000000
        min_mse_iter = 0
        for i in range( 0, self.opts.train_iter + 1 ):
            in_sample, in_label = self.opts.data_source.next_batch()
            # print(in_label)
            self.sess.run( self.optim_nn, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            # for p in range( 5 ):
            #     self.sess.run( self.optim_gate, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            if i % 100 == 0:
                nn_loss = self.sess.run( self.classifier_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                # gate_loss = self.sess.run( self.gate_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                print( "iter: ", i, "NN LOSS: ", nn_loss )
                print("-----------------")
            if i % 1000 == 0:
                self.calc_sparsity( in_sample, in_label )
                in_sample, in_label = self.opts.data_source.get_test()
                mse = self.predict( in_sample, in_label )
                if mse < min_mse:
                    min_mse = mse 
                    min_mse_iter = i
                print( "Iter: ", i, "MSE: ", mse, "MIN MSE: ", min_mse, "MIN MSE Iter: ", min_mse_iter )
                print("-------------------------------------")
                self.mse_list.append( mse )

                np.save( self.opts.cpt_path + "/mse.npy", np.array( self.mse_list ) )
                np.save( self.opts.cpt_path + "/feat_gating.npy", np.array( self.feat_gating_list ) )


            if i != 0 and i % 20000 == 0:
                path = self.opts.cpt_path +"/"+ str( i )
                os.mkdir( path )
                path += "/model.ckpt"
                self.saver.save( self.sess, path )

    def predict( self, sample, label ):
        res = self.sess.run( self.prediction, feed_dict = { self.in_sample : sample, self.in_label: label } )
        accu = np.mean( np.square( res - label ) )

        return accu




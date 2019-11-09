import tensorflow as tf 
import numpy as np 

import tensorflow as tf 
import numpy as np 
import os
from model.tf_l0_models import *

class New_Start_L0_Mnist( object ):
    
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
            self.l0_obj = L0_Dense( 784, 300, activation = tf.nn.leaky_relu, weight_decay = 1., lamba = 1/50000 , name = "l0" )
            self.l0_train, self.l0_test, self.l0_regu = self.l0_obj.build( self.in_sample, self.in_sample )
            
            self.l1_obj = L0_Dense( 300, 100, activation = tf.nn.leaky_relu, weight_decay = 1., lamba = 1/50000, name = "l1" )
            self.l1_train, self.l1_test, self.l1_regu = self.l1_obj.build( self.l0_train, self.l0_test )


            self.logit_obj = L0_Dense( 100, 10, weight_decay = 1./10, lamba = 1/50000., name = "logit" )
            self.logit_train, self.logit_test, self.logit_regu =  self.logit_obj.build( self.l1_train, self.l1_test )
            self.prediction = tf.nn.softmax( self.logit_test )
    
    def construct_loss( self ):
        self.classifier_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits = self.logit_train, labels = self.in_label ) )
        self.classifier_loss = self.classifier_loss \
                        -0.5*( tf.reduce_mean( self.l0_regu ) + tf.reduce_mean( self.l1_regu ) )
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim_nn = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.classifier_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') )

    def calc_sparsity( self, in_sample, in_label, thresh = 0.95 ):
        l0_feat_gate_test, l1_feat_gate_test = self.sess.run( [ self.l0_obj.mask_test, self.l1_obj.mask_test ], 
                                                              feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
        l0_feat_gate_test = l0_feat_gate_test.ravel()
        l1_feat_gate_test = l1_feat_gate_test.ravel()
        print( l0_feat_gate_test[ :10 ] )
        l0 = np.sum( l0_feat_gate_test <= 0.01  )
        l1 = np.sum( l1_feat_gate_test <= 0.01 )

        print( "l0 gate", l0, "out of", l0_feat_gate_test.shape[0], "l1_gate", l1, "out of", l1_feat_gate_test.shape )

    def train( self ):
        self.loss_list = []
        self.accu_list = []
        max_accu = 0
        max_accu_iter = 0
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
                in_sample, in_label = self.opts.data_source.get_test()
                self.calc_sparsity( in_sample, in_label )
                accu = self.predict( in_sample, in_label )
                if accu > max_accu:
                    max_accu = accu 
                    max_accu_iter = i
                print( "Iter: ", i, "Accu: ", accu, "Max Accu: ", max_accu, "Max Accu Iter: ", max_accu_iter )
                print("-------------------------------------")
                self.accu_list.append( accu )


            if i != 0 and i % 20000 == 0:
                path = self.opts.cpt_path +"/"+ str( i )
                os.mkdir( path )
                path += "/model.ckpt"
                self.saver.save( self.sess, path )

    def predict( self, sample, label ):
        res = self.sess.run( self.prediction, feed_dict = { self.in_sample : sample, self.in_label: label } )
        res = np.argmax( res, axis = 1 )
        true = np.argmax( label, axis = 1 )
        print( res[:10] )
        print( true[:10])
        accu = np.sum(res == true) / res.shape[0]

        return accu




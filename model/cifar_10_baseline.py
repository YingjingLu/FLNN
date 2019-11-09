import tensorflow as tf 
import numpy as np 
import math
import tensorflow as tf 
import numpy as np 
import os
from model.new_start_models import *

class Cifar_10_Baseline( object ):
    
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
        self.in_sample = tf.placeholder( tf.float32, [ None ] + [ 3072 ] )
        self.in_label = tf.placeholder( tf.float32, [ None ] + [ 2 ] )

    def construct_clf( self ):
        
        with tf.variable_scope( "nn" ) as scope:
            self.l0, self.l0_pre, self.w_l0, self.b_l0 = dense( self.in_sample, 2048, 
                                                                activation = tf.nn.relu,
                                                                initializer = tf.random_normal_initializer( stddev = 1./tf.sqrt( 2048./2. ) ), 
                                                                name = "l0" )
            self.l1, self.l1_pre, self.w_l1, self.b_l1 = dense( self.l0, 1024, 
                                                                activation = tf.nn.relu,
                                                                initializer = tf.random_normal_initializer( stddev = 1./tf.sqrt( 1024./2. ) ), 
                                                                name = "l1" )
            self.logit_test, _, _, _ =  dense( self.l1, 2, initializer = tf.random_normal_initializer( stddev = 1./tf.sqrt( 2./2. ) ), name = "logit", reuse = False )
            self.prediction = tf.nn.softmax( self.logit_test )
    
    def construct_loss( self ):
        self.classifier_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits = self.logit_test, labels = self.in_label ) )
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim_nn = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.classifier_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') )

    def train( self ):
        self.loss_list = []
        self.accu_list = []
        self.feat_gating_list = []
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
                accu = self.predict( in_sample, in_label )
                if accu > max_accu:
                    max_accu = accu 
                    max_accu_iter = i
                print( "Iter: ", i, "Accu: ", accu, "Max Accu: ", max_accu, "Max Accu Iter: ", max_accu_iter )
                print("-------------------------------------")
                self.accu_list.append( accu )

                np.save( self.opts.cpt_path + "/accu.npy", np.array( self.accu_list ) )


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
    def get_predict( self, sample, label ):
        res = self.sess.run( self.prediction, feed_dict = { self.in_sample : sample, self.in_label: label } )
        np.save( self.opts.cpt_path + "/true.npy", label )
        np.save( self.opts.cpt_path + "/pred.npy", res )




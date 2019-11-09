import tensorflow as tf 
import numpy as np 

import tensorflow as tf 
import numpy as np 
import os

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class New_Start_Invariant_Mnist( object ):
    
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
        with tf.variable_scope( "nn" ) as scope:
            self.in_sample = tf.placeholder( tf.float32, [ None ] + opts.sample_shape )
            self.in_label = tf.placeholder( tf.float32, [ None ] + opts.label_shape )

    def construct_clf( self ):
        with tf.variable_scope( "gating" ) as scope:
            self.v_0_net = tf.Variable(tf.random_normal( [1,784] ), True, name = "v_0_net", dtype=tf.float32)
            self.v_0_feat = tf.Variable(tf.random_normal( [1,784] ), True, name = "v_0_feat", dtype=tf.float32)
            self.g_0_net = tf.nn.sigmoid( self.v_0_net )
            self.g_0_feat = tf.nn.sigmoid( self.v_0_feat )

            self.v_1_net = tf.Variable(tf.random_normal( [1,784 + 256] ), True, name = "v_1_net", dtype=tf.float32)
            self.v_1_feat = tf.Variable(tf.random_normal( [1,784 + 256] ), True, name = "v_1_feat", dtype=tf.float32)
            self.g_1_net = tf.nn.sigmoid( self.v_1_net )
            self.g_1_feat = tf.nn.sigmoid( self.v_1_feat )

            self.v_2_net = tf.Variable(tf.random_normal( [1,784 + 256 + 256] ), True, name = "v_2_net", dtype=tf.float32)
            self.v_2_feat = tf.Variable(tf.random_normal( [1,784 + 256 + 256] ), True, name = "v_2_feat", dtype=tf.float32)
            self.g_2_net = tf.nn.sigmoid( self.v_2_net )
            self.g_2_feat = tf.nn.sigmoid( self.v_2_feat )

        self.l_0_net = self.in_sample * self.g_0_net 
        self.l_0_feat = self.in_sample * self.g_0_feat

        with tf.variable_scope( "nn" ) as scope:
            self.l_1_net_out = tf.layers.dense( self.l_0_net, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.relu, name = "l_1_net_out" )
        self.l_1 = tf.concat( [ self.l_1_net_out, self.l_0_feat ], axis = 1 )
        self.l_1_net = self.l_1 * self.g_1_net 
        self.l_1_feat = self.l_1 * self.g_1_feat 

        with tf.variable_scope( "nn" ) as scope:
            self.l_2_net_out = tf.layers.dense( self.l_1_net, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.relu, name = "l_2_net_out" )
        self.l_2 = tf.concat( [ self.l_2_net_out, self.l_1_feat ], axis = 1 )
        self.l_2_net = self.l_2 * self.g_2_net 
        self.l_2_feat = self.l_2 * self.g_2_feat

        with tf.variable_scope( "nn" ) as scope:
            self.logit = tf.layers.dense( self.l_2_net, 10, name = "dense_out" )
        self.prediction = tf.nn.softmax( self.logit )

    def construct_loss( self ):
        self.nn_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits = self.logit, labels = self.in_label ) )
        # self.gate_loss = self.nn_loss + \
        #                  ( tf.reduce_mean( self.g_0_net ) \
        #                   + tf.reduce_mean( self.g_0_feat ) \
        #                   + tf.reduce_mean( self.g_1_net ) \
        #                   + tf.reduce_mean( self.g_1_feat ) \
        #                   + tf.reduce_mean( self.g_2_net ) ) \
        #                 +(
        #                     tf.reduce_mean( self.g_0_net * self.g_0_feat ) \
        #                     + tf.reduce_mean( self.g_1_net * self.g_1_feat )
        #                 )
        self.gate_loss = self.nn_loss \
                        -(
                            tf.reduce_mean( tf.abs( self.g_0_net - self.g_0_feat ) )\
                            + tf.reduce_mean( tf.abs( self.g_1_net - self.g_1_feat ) )
                        )
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim_nn = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.nn_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') )
        self.optim_gate = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.gate_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gating') )
        # self.optim = tf.train.GradientDescentOptimizer( self.opts.lr ).minimize( loss = self.loss )


    def train( self ):
        self.loss_list = []
        self.accu_list = []
        max_accu = 0
        max_accu_iter = 0
        print("in training")
        for i in range( 0, self.opts.train_iter + 1 ):
            in_sample, in_label = self.opts.data_source.next_batch()
            # print(i)
            self.sess.run( self.optim_nn, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            for p in range(5):
                self.sess.run( self.optim_gate, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            if i % 100 == 0:
                nn_loss = self.sess.run( self.nn_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                gate_loss = self.sess.run( self.gate_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                print( "iter: ", i, "NN LOSS: ", nn_loss, "Gate LOSS: ", gate_loss )
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




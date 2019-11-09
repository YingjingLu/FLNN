import tensorflow as tf 
import numpy as np 

import tensorflow as tf 
import numpy as np 
import os

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    res = mu + tf.exp(log_var/2)*eps
    return res

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

def get_z_sample( batch_size, z_dim ):
    return tf.random_normal(tf.stack([batch_size, z_dim]))

class AE_Stochastic_Mnist( object ):
    
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
            self.lc0 = tf.layers.dense( self.in_sample, 784, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.sigmoid, name = "gating_0" )
        self.l = self.in_sample #* self.lc0
        
        with tf.variable_scope( "nn" ) as scope:
            self.l0g0 = tf.layers.dense( self.l, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.relu, name = "dense_1" )
        with tf.variable_scope( "gating" ) as scope:
            self.l0c0 = tf.layers.dense( self.l, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.sigmoid, name = "gating_1" )
        self.l0 = self.l0g0 * self.l0c0

        with tf.variable_scope( "nn" ) as scope:
            self.l1g0 = tf.layers.dense( self.l0, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.relu, name = "dense_2" )
        with tf.variable_scope( "gating" ) as scope:
            self.l1c0 = tf.layers.dense( self.l0, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.sigmoid, name = "gating_2" )
        self.l1 = self.l1g0 * self.l1c0

        with tf.variable_scope( "nn" ) as scope:
            self.logit = tf.layers.dense( self.l1, 10, name = "dense_out" )
        self.prediction = tf.nn.softmax( self.logit )
        with tf.variable_scope( "decode" ) as scope:
            self.mu = tf.layers.dense( self.l1, 100,  kernel_initializer=tf.random_normal_initializer, activation = None, name = "mu" )
            self.logvar = tf.layers.dense( self.l1, 100, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.softplus, name = "logvar" )
            self.z = sample_z( self.mu, self.logvar )
            self.l2 = tf.layers.dense( self.l1, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.relu, name = "dense_3" )
            self.l3 = tf.layers.dense( self.l2, 256, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.relu, name = "dense_4" )
            self.l4 = self.l2 = tf.layers.dense( self.l3, 784, kernel_initializer=tf.random_normal_initializer, activation = tf.nn.tanh, name = "dense_5" )
    
    def construct_loss( self ):
        self.classifier_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits = self.logit, labels = self.in_label ) )
        self.nn_loss = self.classifier_loss # + tf.reduce_mean( tf.abs( self.l0g ) ) + tf.reduce_mean( tf.abs( self.l1g ) ) + tf.reduce_mean( tf.abs( self.logit ) )
        self.gate_loss = self.classifier_loss + ( tf.reduce_mean( self.l0c0 ) + tf.reduce_mean( self.l1c0 ) )
        
        self.elbo = 0.5 * tf.reduce_sum(tf.exp(self.logvar) + self.mu**2 - 1. - self.logvar, 1)
        self.mmd = compute_mmd(get_z_sample( self.opts.batch_size, 100 ), self.z)
        self.recon = tf.reduce_sum( tf.square( self.in_sample - self.l4 ), axis = 1 )
        self.vae_loss = tf.reduce_mean( self.recon)
        
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim_nn = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.nn_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') )
        self.optim_gate = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.gate_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gating') )
        # self.optim_vae = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.vae_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') + \
        #                                                                                                                                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decode') )
        # self.optim = tf.train.GradientDescentOptimizer( self.opts.lr ).minimize( loss = self.loss )


    def train( self ):
        self.loss_list = []
        self.accu_list = []
        for i in range( 0, self.opts.train_iter + 1 ):
            in_sample, in_label = self.opts.data_source.next_batch()
            # print(in_label)
            # self.sess.run( self.optim_vae, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            self.sess.run( self.optim_nn, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            self.sess.run( self.optim_gate, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            if i % 100 == 0:
                nn_loss = self.sess.run( self.nn_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                gate_loss = self.sess.run( self.gate_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                print( "iter: ", i, "NN LOSS: ", nn_loss, "Gate LOSS: ", gate_loss )
                print("-----------------")
            if i % 1000 == 0:
                in_sample, in_label = self.opts.data_source.get_test()
                accu = self.predict( in_sample, in_label )
                print( "Iter: ", i, "Accu: ", accu )
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




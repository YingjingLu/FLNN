import tensorflow as tf 
import numpy as np 
import os

class Baseline_Mnist( object ):
    
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
        self.l0 = tf.layers.dense( self.in_sample, 300, kernel_initializer=tf.keras.initializers.he_normal(), activation = tf.nn.leaky_relu, name = "dense_1" )
        # self.l1 = tf.layers.dense( self.l0, 512, kernel_initializer=tf.keras.initializers.he_normal(), activation = tf.nn.relu, name = "dense_2" )
        self.logit = tf.layers.dense( self.l0, 10, kernel_initializer=tf.keras.initializers.he_normal(), name = "dense_out" )
        self.prediction = tf.nn.softmax( self.logit )

    def construct_loss( self ):
        self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits = self.logit, labels = self.in_label ) )
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.loss )
        # self.optim = tf.train.GradientDescentOptimizer( self.opts.lr ).minimize( loss = self.loss )


    def train( self ):
        self.loss_list = []
        self.accu_list = []
        for i in range( self.opts.train_iter + 1 ):
            in_sample, in_label = self.opts.data_source.next_batch()
            # print(in_label)
            self.sess.run( self.optim, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
            if i % 100 == 0:
                loss = self.sess.run( self.loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                print( "iter: ", i, "LOSS: ", loss )
                self.loss_list.append( loss )
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




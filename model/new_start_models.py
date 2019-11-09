import numpy as np 
import tensorflow as tf 
import math 

########################## Globals #################################
EPSILON = 1e-6
LIMIT_A = -0.1
LIMIT_B = 1.1
zero = lambda: tf.constant(0.)
one = lambda: tf.constant(1.)
########################## utilities ###############################

def binary_activation(x):

    cond = tf.less(x, tf.zeros(tf.shape(x)) + 1e-3)
    out = tf.where(cond, tf.ones(tf.shape(x)), tf.zeros(tf.shape(x)))

    return out

def hardtanh( inputs, min_val = -1.0, max_val = 1.0 ):
    return tf.maximum( min_val, tf.minimum( max_val, inputs ) )

def dense( inputs, h_size, 
              bias = True, initializer = tf.random_normal_initializer, activation = None, reuse = False,
              name = None ):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope( name or "Linear", reuse = reuse ) as scope:
        w = tf.get_variable( "w", [ shape[1], h_size ], tf.float32, initializer = initializer )
        b = None
        if bias:
            b = tf.get_variable( "b", [ h_size ], initializer = tf.constant_initializer( 0.0 ) )

        if bias:
            pre_act = tf.matmul( inputs, w ) + bias
        else:
            pre_act = tf.matmul( inputs, w )
        if activation is not None:
            return activation( pre_act ), pre_act, w, b 
        else:
            return pre_act, pre_act, w, b 

class Conv( object ):
    def __init__( self, name ):
        self.name = name 

    def __call__( self, *args ):
        print( "HAHA" )

    def __repr__( self ):
        return self.name 

    def get_weights( self, sess ):
        assert ( self.w is not None ) and ( self. b is not None ), "weights in the conv layer should be initialized"

        w = sess.run( self.w )
        b = sess.run( self.b )

        return w, b


class Conv2D( Conv ):

    def __init__( self, filters = 8, \
                  kernel_w = 3, kernel_h= 3, 
                  stride_h = 1, stride_v = 1,
                  padding = "VALID",
                  activation = None,
                  use_bias = True,
                  kernel_init = tf.glorot_normal_initializer, bias_init = tf.zeros_initializer,
                  kernel_regu = None, bias_regu = None,
                  activation_regu = None,
                  bias_constraint = None,
                  reuse = tf.AUTO_REUSE,
                  trainable = True,
                  name = "Conv2D" ):

        super( Conv2D, self ).__init__( name )
        self.filters = filters
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.stride_h = stride_h
        self.stride_v = stride_v
        self.padding = padding
        self.activation = activation 
        self.use_bias = use_bias
        self.kernel_init = kernel_init 
        self.bias_init = bias_init 
        self.kernel_regu = kernel_regu
        self.bias_regu = bias_regu 
        self.activation_regu = activation_regu
        self.bias_constraint = bias_constraint
        self.reuse = reuse
        self.trainable = trainable


    def __call__( self, x ):
        self.input_shape = shape = x.get_shape().as_list()
        with tf.variable_scope( self.name, reuse = self.reuse ) as scope:
            self.w = tf.get_variable( shape = [ self.kernel_h, self.kernel_w, shape[ -1 ], self.filters ],
                                  initializer = self.kernel_init,
                                  regularizer = self.kernel_regu,
                                  trainable = self.trainable, name = self.name + "_kernels" )
            if self.use_bias:
                self.b = tf.get_variable( shape = [ self.filters ],
                                      initializer = self.bias_init,
                                      regularizer = self.bias_regu,
                                      trainable = self.trainable, name = self.name + "_bias" )
            conv = tf.nn.conv2d( x, 
                                self.w,
                                strides = [ 1, self.stride_h, self.stride_v, 1 ], 
                                padding = self.padding )
            if self.use_bias:
                conv = tf.nn.bias_add( conv, self.b )
            if self.activation is None:
                return conv
            else:
                return self.activation( conv )

class Group_L0( object ):
    
    def __init__( self, in_size, out_size, activation = None,
                  bias = True, weight_decay = 1.0, drop_rate_init = 0.5, temp= 2./3., 
                  lamba = 1., local_rep = False, name = None, **kwargs ):


        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.prior_prec = weight_decay
        self.temp = temp
        self.droprate_init = drop_rate_init
        self.lamba = lamba 
        self.use_bias = False
        self.local_rep = local_rep
        with tf.variable_scope( name or "Group_L0", reuse = tf.AUTO_REUSE ) as scope:
            self.w = tf.get_variable( "w", [ in_size, out_size ], tf.float32, 
                                      initializer = tf.random_normal_initializer( stddev = 1. / math.sqrt( out_size / 2. ) ) )
            self.qz_loga = tf.get_variable( "qz_loga", [ 1, in_size ], tf.float32,
                                            initializer = tf.random_normal_initializer( stddev = 1. / math.sqrt( out_size / 2. ) ),
                                            constraint = lambda t: tf.clip_by_value(t, math.log( 1e-2 ), math.log( 1e2 ) ) )
            if bias:
                self.b = tf.get_variable( "b", [ out_size ], tf.float32,
                                          initializer = tf.constant_initializer( 0.0 ) )
                self.use_bias = True 
    
    def get_eps( self, inputs ):
        shape = inputs.get_shape().as_list()
        return tf.random_uniform( shape, EPSILON, 1.0-EPSILON ) 

    def quartile_concrete( self, x ):
        y = tf.nn.sigmoid( ( tf.log( x ) - tf.log( 1-x ) + self.qz_loga ) / self.temp )
        return y * ( LIMIT_B - LIMIT_A ) + LIMIT_A  


    def sample_z( self, inputs, sample = True ):
        shape = inputs.get_shape().as_list()
        if sample:
            eps = self.get_eps( inputs )
            z = self.quartile_concrete( eps )
            return hardtanh( z, min_val = 0.0, max_val = 1.0 )
        else:
            pi = tf.nn.sigmoid( self.qz_loga )
            return hardtanh( pi * ( LIMIT_B - LIMIT_A ) + LIMIT_A, min_val = 0.0, max_val = 1.0 )
    
    def sample_masks( self ):
        z = self.get_eps( tf.zeros( [ 1, self.in_size ] ) )
        z = self.quartile_concrete( z )
        z = tf.transpose( z )
        net_mask = hardtanh( z, min_val = 0.0, max_val = 1.0 )
        # mask = tf.broadcast_to( mask, [ self.in_size, self.out_size ] )
        return tf.transpose( 1.-net_mask ), net_mask

    def cdf_qz( self, x ):
        xn = ( x - LIMIT_A ) / ( LIMIT_B - LIMIT_A )
        logits = math.log( xn ) - math.log( 1. - xn )
        res = tf.nn.sigmoid( logits * self.temp - self.qz_loga )
        res = tf.clip_by_value( res, clip_value_min = EPSILON, clip_value_max = 1. - EPSILON )
        return res

    def regu( self ):
        logpw_col = tf.reduce_mean( -1. * ( .5 * self.prior_prec * tf.math.pow( self.w, 2. ) ) - self.lamba, axis = 1 )
        log_pw = tf.reduce_mean( ( 1 - self.cdf_qz( 0 ) ) * logpw_col )
        logpb = 0. if not self.use_bias else -1.0 * tf.reduce_mean( 0.5 * self.prior_prec * tf.math.pow( self.b, 2 ) )
        return log_pw + logpb

    def build( self, train_inputs, test_inputs ):
        

        # case for not training
        self.net_mask_test = self.sample_z( test_inputs, sample = False )
        c = tf.zeros_like( self.net_mask_test )
        zero = lambda: c

        c = tf.ones_like( self.net_mask_test )
        one = lambda: c
        # self.feature_mask_test = 1.0 - tf.maximum( self.net_mask_test, 1 )
        self.feature_mask_test = binary_activation( self.net_mask_test )
        self.xin = tf.math.multiply( test_inputs, self.net_mask_test )
        self.net_output_test = tf.linalg.matmul( self.xin, self.w )
        self.feature_output_test = test_inputs * self.feature_mask_test

        # case for trinaning
        self.feature_mask_train, self.net_mask_train = self.sample_masks()
        self.feature_output_train = train_inputs * self.feature_mask_train
        self.net_output_train = tf.linalg.matmul( train_inputs, self.w * self.net_mask_train )


        if self.use_bias:
            self.net_output_test += self.b 
            self.net_output_train += self.b 

        self.regularization = self.regu()
        if self.activation is None:
            return ( self.net_output_train, self.feature_output_train, 
                     self.net_output_test, self.feature_output_test, 
                     self.regularization )
        else:
            return ( self.activation( self.net_output_train ), self.activation( self.feature_output_train ), 
                     self.activation( self.net_output_test ), self.activation( self.feature_output_test ), 
                     self.regularization )

class Conv_L0( object ):
    
    def __init__( self, in_channel, out_channel, in_h, in_w, kernel_w = 3, kernel_h= 3, 
                  stride_h = 1, stride_v = 1,
                  bias = True,
                  padding = "VALID", activation = None,
                  kernel_init = tf.glorot_normal_initializer, bias_init = tf.zeros_initializer,
                  kernel_regu = None, bias_regu = None,
                   weight_decay = 1.0, drop_rate_init = 0.5, temp= 2./3., 
                  lamba = 1., local_rep = False, name = None, **kwargs ):


        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_h = in_h 
        self.in_w = in_w
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.stride_h = stride_h
        self.stride_v = stride_v
        self.padding = padding
        self.activation = activation
        self.prior_prec = weight_decay
        self.temp = temp
        self.droprate_init = drop_rate_init
        self.lamba = lamba 
        self.use_bias = False
        self.local_rep = local_rep
        with tf.variable_scope( name or "Group_L0", reuse = tf.AUTO_REUSE ) as scope:
            self.w = tf.get_variable( shape = [ kernel_h, kernel_w, in_channel, out_channel ],
                                  initializer = kernel_init,
                                  regularizer = kernel_regu,
                                  trainable = True, name = name + "_kernels" )
            if bias:
                self.b = tf.get_variable( shape = [ out_channel ],
                                      initializer = bias_init,
                                      regularizer = bias_regu,
                                      trainable = True, name = name + "_bias" )
                self.use_bias = True 

            self.qz_loga = tf.get_variable( "qz_loga", [ in_h, in_w, in_channel ], tf.float32,
                                            initializer = kernel_init,
                                            constraint = lambda t: tf.clip_by_value(t, math.log( 1e-2 ), math.log( 1e2 ) ) )
            
    
    def get_eps( self, inputs ):
        shape = inputs.get_shape().as_list()
        return tf.random_uniform( shape, EPSILON, 1.0-EPSILON ) 

    def quartile_concrete( self, x ):
        y = tf.nn.sigmoid( ( tf.log( x ) - tf.log( 1-x ) + self.qz_loga ) / self.temp )
        return y * ( LIMIT_B - LIMIT_A ) + LIMIT_A  


    def sample_z( self, inputs, sample = True ):
        if sample:
            eps = self.get_eps( inputs )
            z = self.quartile_concrete( eps )
            return hardtanh( z, min_val = 0.0, max_val = 1.0 )
        else:
            pi = tf.nn.sigmoid( self.qz_loga )
            return hardtanh( pi * ( LIMIT_B - LIMIT_A ) + LIMIT_A, min_val = 0.0, max_val = 1.0 )
    
    def sample_masks( self ):
        z = self.get_eps( tf.zeros( [ self.in_h, self.in_w, self.in_channel ] ) )
        z = self.quartile_concrete( z )
        net_mask = hardtanh( z, min_val = 0.0, max_val = 1.0 )
        # mask = tf.broadcast_to( mask, [ self.in_size, self.out_size ] )
        return 1.-net_mask, net_mask

    def cdf_qz( self, x ):
        xn = ( x - LIMIT_A ) / ( LIMIT_B - LIMIT_A )
        logits = math.log( xn ) - math.log( 1. - xn )
        res = tf.nn.sigmoid( logits * self.temp - self.qz_loga )
        res = tf.clip_by_value( res, clip_value_min = EPSILON, clip_value_max = 1. - EPSILON )
        return res

    def regu( self ):
        logpw_col = tf.reduce_mean( -1. * ( .5 * self.prior_prec * tf.math.pow( self.w, 2. ) ) - self.lamba, axis = [0,1,3] )
        log_pw = tf.reduce_mean( tf.reduce_mean( ( 1 - self.cdf_qz( 0 ) ), axis = [0,1] ) * logpw_col )
        logpb = 0. if not self.use_bias else -1.0 * tf.reduce_mean( 0.5 * self.prior_prec * tf.math.pow( self.b, 2 ) )
        return log_pw + logpb

    def build( self, train_inputs, test_inputs ):
        

        # case for not training
        self.net_mask_test = self.sample_z( test_inputs, sample = False )
        c = tf.zeros_like( self.net_mask_test )
        zero = lambda: c

        c = tf.ones_like( self.net_mask_test )
        one = lambda: c
        # self.feature_mask_test = 1.0 - tf.maximum( self.net_mask_test, 1 )
        self.feature_mask_test = binary_activation( self.net_mask_test )
        self.xin = tf.math.multiply( test_inputs, self.net_mask_test )

        self.net_output_test = tf.nn.conv2d( self.xin, 
                                self.w,
                                strides = [ 1, self.stride_h, self.stride_v, 1 ], 
                                padding = self.padding )
        if self.use_bias:
            self.net_output_test = tf.nn.bias_add( self.net_output_test, self.b )
        self.feature_output_test = test_inputs * self.feature_mask_test
        # case for trinaning
        self.feature_mask_train, self.net_mask_train = self.sample_masks()
        self.feature_output_train = train_inputs * self.feature_mask_train
        self.net_output_train = tf.nn.conv2d( train_inputs* self.net_mask_train, 
                                self.w ,
                                strides = [ 1, self.stride_h, self.stride_v, 1 ], 
                                padding = self.padding )
        if self.use_bias:
            self.net_output_train = tf.nn.bias_add( self.net_output_train, self.b )


        if self.use_bias:
            self.net_output_test = tf.nn.bias_add( self.net_output_test, self.b )
            self.net_output_train = tf.nn.bias_add( self.net_output_train, self.b )

        self.regularization = self.regu()
        if self.activation is None:
            return ( self.net_output_train, self.feature_output_train, 
                     self.net_output_test, self.feature_output_test, 
                     self.regularization )
        else:
            return ( self.activation( self.net_output_train ), self.activation( self.feature_output_train ), 
                     self.activation( self.net_output_test ), self.activation( self.feature_output_test ), 
                     self.regularization )
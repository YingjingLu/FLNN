import tensorflow as tf 
import numpy as np 

import tensorflow as tf 
import torch
import numpy as np 
import torchvision
import torchvision.transforms as transforms
import os
from model.new_start_models import *

class L0_Cifar( object ):
    
    def __init__(self, opts, sess):
        self.sess = sess
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
        self.in_sample = tf.placeholder( tf.float32, [ None ] + [ 32, 32, 3 ] )
        self.in_label = tf.placeholder( tf.int32, [ None ] )

    def construct_clf( self ):
        
        with tf.variable_scope( "nn" ) as scope:
            self.l0_obj = Conv_L0( 3, 64, 32, 32, stride_h = 2, stride_v = 2, padding = "SAME", activation = tf.nn.leaky_relu, weight_decay = 1.0 , name = "l0" )
            self.net_l0_train, self.feat_l0_train, self.net_l0_test, self.feat_l0_test, self.l0_regu = self.l0_obj.build( self.in_sample, self.in_sample )

            self.l00_obj = Conv2D( 64, stride_h = 2, stride_v = 2, padding = "SAME", activation = None, name = "l00" )
            self.feat_l00_train, self.feat_l00_test = self.l00_obj( self.feat_l0_train ), self.l00_obj( self.feat_l0_test )

            self.l01_obj = Conv2D( 64, stride_h = 2, stride_v = 2, padding = "SAME", activation = None, name = "l01" )
            self.feat_l01_train, self.feat_l01_test = self.l01_obj( self.feat_l00_train ), self.l01_obj( self.feat_l00_test )

            self.l02_obj = Conv2D( 64, stride_h = 2, stride_v = 2, padding = "SAME", activation = None, name = "l02" )
            self.feat_l02_train, self.feat_l02_test = self.l02_obj( self.feat_l01_train ), self.l02_obj( self.feat_l01_test )



            self.l1_obj = Conv_L0( 64, 128, 16, 16, stride_h = 2, stride_v = 2, padding = "SAME", activation = tf.nn.leaky_relu, weight_decay = 1.0 , name = "l1" )
            self.net_l1_train, self.feat_l1_train, self.net_l1_test, self.feat_l1_test, self.l1_regu = self.l1_obj.build( self.net_l0_train, self.net_l0_test )

            self.l10_obj = Conv2D( 128, stride_h = 2, stride_v = 2, padding = "SAME", activation = None, name = "l10" )
            self.feat_l10_train, self.feat_l10_test = self.l10_obj( self.feat_l1_train ), self.l10_obj( self.feat_l1_test )

            self.l11_obj = Conv2D( 128, stride_h = 2, stride_v = 2, padding = "SAME", activation = None, name = "l11" )
            self.feat_l11_train, self.feat_l11_test = self.l11_obj( self.feat_l10_train ), self.l11_obj( self.feat_l10_test )



            
            self.l2_obj = Conv_L0( 128, 256, 8, 8, stride_h = 2, stride_v = 2, padding = "SAME", activation = tf.nn.leaky_relu, weight_decay = 1.0 , name = "l2" )
            self.net_l2_train, self.feat_l2_train, self.net_l2_test, self.feat_l2_test, self.l2_regu = self.l2_obj.build( self.net_l1_train, self.net_l1_test )

            self.l20_obj = Conv2D( 256, stride_h = 2, stride_v = 2, padding = "SAME", activation = None, name = "l20" )
            self.feat_l20_train, self.feat_l20_test = self.l20_obj( self.feat_l2_train ), self.l20_obj( self.feat_l2_test )



            self.l3_obj = Conv2D( 256, 4, 4, stride_h = 2,  stride_v = 2, padding = "SAME", activation = tf.nn.leaky_relu, name = "l3" )
            self.net_l3_train, self.net_l3_test = self.l3_obj( self.net_l2_train ), self.l3_obj( self.net_l2_test )

            self.concat_train = tf.concat( [ tf.layers.flatten( self.feat_l02_train ), tf.layers.flatten( self.feat_l11_train ), tf.layers.flatten( self.feat_l20_train ), tf.layers.flatten( self.net_l3_train ) ], axis = 1 )
            self.concat_test = tf.concat( [ tf.layers.flatten( self.feat_l02_test ), tf.layers.flatten( self.feat_l11_test ), tf.layers.flatten( self.feat_l20_test ), tf.layers.flatten( self.net_l3_test ) ], axis = 1 )
            # self.logit_obj = Group_L0( 784 + 300 + 100, 10, weight_decay = 5e-4, name = "logit" )
            self.logit_train, _, self.final_w, _ =  dense( self.concat_train, 10, initializer = tf.glorot_normal_initializer, name = "logit" )
            self.logit_test, _, _, _ =  dense( self.concat_test, 10, initializer = tf.glorot_normal_initializer, name = "logit", reuse = True )
            self.prediction = tf.nn.softmax( self.logit_test )
    
    def construct_loss( self ):
        self.classifier_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.logit_train, labels = self.in_label ) )
        self.classifier_loss = self.classifier_loss \
                        -0.1*( tf.reduce_mean( self.l0_regu ) + tf.reduce_mean( self.l1_regu ) + tf.reduce_mean( self.l2_regu ) )
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim_nn = tf.train.AdamOptimizer( self.opts.lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.classifier_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') )

    def calc_sparsity( self, in_sample, in_label, thresh = 0.999 ):
        l0_feat_gate_test, l1_feat_gate_test, l2_feat_gate_test = self.sess.run( [ self.l0_obj.feature_mask_test, self.l1_obj.feature_mask_test, self.l2_obj.feature_mask_test ], 
                                                                                feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
        l0_feat_gate_test = l0_feat_gate_test.ravel()
        l1_feat_gate_test = l1_feat_gate_test.ravel()
        l2_feat_gate_test = l2_feat_gate_test.ravel()
        print( l0_feat_gate_test[ :10 ] )
        l0 = np.sum( l0_feat_gate_test >= thresh )
        l1 = np.sum( l1_feat_gate_test >= thresh )
        l2 = np.sum( l2_feat_gate_test >= thresh )

        print( "l0 gate", l0, "out of", l0_feat_gate_test.shape[0], "l1_gate", l1, "out of", l1_feat_gate_test.shape[0], l2, "out of", l2_feat_gate_test.shape[0] )
        self.feat_gating_list.append( [ l0, l1 ] )

    def train( self ):
        self.loss_list = []
        self.accu_list = []
        self.feat_gating_list = []
        max_accu = 0
        max_accu_iter = 0
        mean = 0.5
        std = 0.5
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill = ( 128, 128, 128 ) ),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=transform_train) #训练数据集
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.opts.batch_size, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

        testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.opts.batch_size, shuffle=False, num_workers=2)

        for epoch in range( 1, 300 ):
            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                in_sample = inputs.numpy().transpose( ( 0, 2, 3, 1 ) )
                in_label = labels.numpy()
                if inputs.shape[ 0 ] == self.opts.batch_size:
                    # print(in_label)
                    self.sess.run( self.optim_nn, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                    # for p in range( 5 ):
                    #     self.sess.run( self.optim_gate, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                    if i % 100 == 0:
                        nn_loss = self.sess.run( self.classifier_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                        # gate_loss = self.sess.run( self.gate_loss, feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
                        print( "iter: ", i, "NN LOSS: ", nn_loss )
                        print("-----------------")
            self.calc_sparsity( in_sample, in_label )
            in_sample = []
            in_label = []
            for j, test_data in enumerate(testloader, 0):
                test_in, test_l = test_data
                test_sample = test_in.numpy().transpose( ( 0, 2, 3, 1 ) )
                test_label = test_l.numpy()
                in_sample.append( test_sample )
                in_label.append( test_l )
            in_sample = np.concatenate( in_sample, 0 )
            in_label = np.concatenate( in_label, 0 )
            accu = self.predict( in_sample, in_label )
            if accu > max_accu:
                max_accu = accu 
                max_accu_iter = epoch
                self.get_predict( in_sample, in_label )
            print( "Iter: ", epoch, "Accu: ", accu, "Max Accu: ", max_accu, "Max Accu Iter: ", max_accu_iter )
            print("-------------------------------------")
            self.accu_list.append( accu )

            np.save( self.opts.cpt_path + "/accu.npy", np.array( self.accu_list ) )
            np.save( self.opts.cpt_path + "/feat_gating.npy", np.array( self.feat_gating_list ) )


            if epoch % 3 == 0 or accu > max_accu:
                path = self.opts.cpt_path +"/"+ str( epoch )
                os.mkdir( path )
                path += "/self.ckpt"
                self.saver.save( self.sess, path )

    def predict( self, sample, label ):
        res = self.sess.run( self.prediction, feed_dict = { self.in_sample : sample, self.in_label: label } )
        res = np.argmax( res, axis = 1 )
        true = label
        print( res[:10] )
        print( true[:10])
        accu = np.sum(res == true) / res.shape[0]

        return accu
    def get_predict( self, sample, label ):
        res = self.sess.run( self.prediction, feed_dict = { self.in_sample : sample, self.in_label: label } )
        np.save( self.opts.cpt_path + "/true.npy", label )
        np.save( self.opts.cpt_path + "/pred.npy", res )




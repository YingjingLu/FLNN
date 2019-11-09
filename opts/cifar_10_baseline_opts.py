import tensorflow as tf 

class Cifar_10_Baseline_Opts( object ):
    def __init__( self ):
        self.batch_size = 32
        self.sample_shape = [ 3072 ]
        self.label_shape = [ 2 ]
        self.num_class = 2


        self.lr = 5e-5
        self.train_alt = 100
        self.train_iter = 2000000

        self.sample_path = "../data/mnist"
        self.cpt_path = "cpt/cifar_10_baseline"
        self.train_sample_path = "../data/cifar-10/train_x.npy"
        self.train_label_path = "../data/cifar-10/train_y.npy"
        self.test_sample_path = "../data/cifar-10/test_x.npy"
        self.test_label_path = "../data/cifar-10/test_y.npy"
        self.train = 1
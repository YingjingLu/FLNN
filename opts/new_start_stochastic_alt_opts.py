import tensorflow as tf 

class New_Start_Stochastic_Alt_Opts( object ):
    def __init__( self ):
        self.batch_size = 64
        self.sample_shape = [ 784 ]
        self.label_shape = [ 10 ]
        self.num_class = 10


        self.lr = 1e-4
        self.train_alt = 100
        self.train_iter = 1000000

        self.sample_path = "../data/mnist"
        self.cpt_path = "cpt/new_start_stochastic_alt_mnist"
        self.train_sample_path = "../data/mnist/train_x.npy"
        self.train_label_path = "../data/mnist/train_y.npy"
        self.test_sample_path = "../data/mnist/test_x.npy"
        self.test_label_path = "../data/mnist/test_y.npy"
        self.train = 0
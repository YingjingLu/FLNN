import tensorflow as tf 

class New_Start_Invariant_Spiral_Opts( object ):
    def __init__( self ):
        self.batch_size = 64
        self.sample_shape = [ 3 ]
        self.label_shape = [ 2 ]
        self.num_class = 2


        self.lr = 1e-4
        self.train_alt = 100
        self.train_iter = 1000000

        self.sample_path = "../data/independent_xor"
        self.cpt_path = "cpt/new_start_invariant_xor_new_8_loss_NoGate"
        self.train_sample_path = "../data/independent_xor/new_xor_x.npy"
        self.train_label_path = "../data/independent_xor/new_xor_y.npy"
        self.test_sample_path = "../data/independent_xor/new_xor_test_x.npy"
        self.test_label_path = "../data/independent_xor/new_xor_test_y.npy"
        self.train = 1
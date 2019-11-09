import tensorflow as tf 

class New_Start_Final_Cal_Opts( object ):
    def __init__( self ):
        self.batch_size = 64
        self.sample_shape = [ 13 ]
        self.label_shape = [ 1 ]
        self.num_class = 1


        self.lr = 1e-4
        self.train_alt = 100
        self.train_iter = 1000000

        self.sample_path = "../data/mnist"
        self.cpt_path = "cpt/new_start_final_cal_housing"
        self.train_sample_path = "../data/cal_housing/train_x.npy"
        self.train_label_path = "../data/cal_housing/train_y.npy"
        self.test_sample_path = "../data/cal_housing/test_x.npy"
        self.test_label_path = "../data/cal_housing/test_y.npy"
        self.train = 1
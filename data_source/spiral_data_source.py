import numpy as np

class Data_Source( object ):

    def __init__( self, opts ):
        self.batch_size = opts.batch_size
        
        self.train_sample = None
        self.test_sample = None 
        self.train_label = None 
        self.test_label = None

        self.num_train = 0
        self.num_test = 0
        self.cur_train = 0
        self.cur_test = 0
        self.opts = opts

    def append_train_sample( self, sample_matrix ):
        if self.train_sample is None:
            self.train_sample = sample_matrix 
        else:
            self.train_sample = np.concatenate( ( self.train_sample, sample_matrix ), axis = 0 )
        self.num_train += sample_matrix.shape[0]
    def append_train_label( self, label_matrix ):
        if self.train_label is None:
            self.train_label = label_matrix 
        else:
            self.train_label = np.concatenate( ( self.train_label, label_matrix ), axis = 0 )

    def append_test_sample( self, sample_matrix ):
        if self.test_sample is None:
            self.test_sample = sample_matrix 
        else:
            self.test_sample = np.concatenate( ( self.test_sample, sample_matrix ), axis = 0 )
        self.num_test += sample_matrix.shape[0]
    def append_test_label( self, label_matrix ):
        if self.test_label is None:
            self.test_label = label_matrix 
        else:
            self.test_label = np.concatenate( ( self.test_label, label_matrix ), axis = 0 )

    def load_unsplit_samples( self ):
        train_sample_path = self.opts.train_sample_path
        test_sample_path = self.opts.test_sample_path
        train_label_path = self.opts.train_label_path
        test_label_path = self.opts.test_label_path
        self.append_train_sample( np.load( train_sample_path ) )
        self.append_test_sample( np.load( test_sample_path) )
        self.append_train_label( np.load( train_label_path ) )
        self.append_test_label( np.load( test_label_path ) )
        # initial shuffle
        index = np.arange( self.num_train, dtype = np.int )
        np.random.shuffle( index )
        self.train_label = self.train_label[ index ]
        self.train_sample = self.train_sample[ index ]

        index = np.arange( self.num_test, dtype = np.int )
        np.random.shuffle( index )
        self.test_label = self.test_label[ index ]
        self.test_sample = self.test_sample[ index ]
        print(self.num_train, self.train_label.shape[0], self.num_test, self.test_label.shape[0])
        if ( self.num_train != self.train_label.shape[0] or self.num_test != self.test_label.shape[0]):
            raise EnvironmentError( "Train or test label or sample does not match" ) 

    def next_batch( self, batch_size = -1 ):
        if batch_size == -1:
            batch_size = self.batch_size
        sample = self.train_sample[ self.cur_train: self.cur_train + batch_size, : ]
        label = self.train_label[ self.cur_train: self.cur_train + batch_size, : ]
        self.cur_train += self.batch_size
        # print(self.cur_train)
        if( self.cur_train + self.batch_size >= self.num_train ):
            index = np.arange( self.num_train, dtype = np.int )
            np.random.shuffle( index )
            self.train_label = self.train_label[ index ]
            self.train_sample = self.train_sample[ index ]
            self.cur_train = 0
        return sample, label

    def get_test( self, batch_size = -1 ):
        # if batch_size == -1:
        #     batch_size = self.batch_size
        # sample = self.test_sample[ self.cur_test: self.cur_test + batch_size, : ]
        # label = self.test_label[ self.cur_test: self.cur_test + batch_size, : ]
        # self.cur_test += batch_size
        # if( self.cur_test + self.batch_size >= self.num_test ):
        #     self.cur_test = 0
        #     return None, None
        if batch_size == -1:
            return self.test_sample, self.test_label
        else:
            return self.test_sample[ :batch_size ], self.test_label[ :batch_size ]
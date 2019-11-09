import gzip f = CLF( opts )
import numpy as np 

def save_mnist():
    with gzip.open( "../../data/mnist/train-images-idx3-ubyte.gz" ) as f:
        m = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        np.save("../../data/mnist/train_x.npy", m)
        print(m.shape)
    with gzip.open( "../../data/mnist/t10k-images-idx3-ubyte.gz" ) as f:
        m = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        np.save("../../data/mnist/test_x.npy", m)
        print(m.shape)
    with gzip.open( "../../data/mnist/t10k-labels-idx1-ubyte.gz" ) as f:
        m = np.frombuffer(f.read(), np.uint8, offset=8)
        print(m)
        l = np.zeros( ( m.shape[0], 10 ) )
        for i in range(m.shape[0]):
            l[ i, m[ i ] ] = 1
        print(l)
        np.save("../../data/mnist/test_y.npy", l)
        print(m.shape)
    with gzip.open( "../../data/mnist/train-labels-idx1-ubyte.gz" ) as f:
        m = np.frombuffer(f.read(), np.uint8, offset=8)
        l = np.zeros( ( m.shape[0], 10 ) )
        for i in range(m.shape[0]):
            l[ i, m[ i ] ] = 1
        np.save("../../data/mnist/train_y.npy", l)
        print(m.shape)

save_mnist()
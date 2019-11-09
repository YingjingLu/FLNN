import numpy as np 

def generate( num_sample = 1000, thresh = 0.5 ):
    x = np.random.uniform( -1, 1, size = num_sample )
    y = np.random.uniform( -1, 1, size = num_sample )

    noise_x = np.random.normal(0., 0.2, size = num_sample )
    noise_y = np.random.normal(0., 0.2, size = num_sample )

    x += noise_x
    y += noise_y

    scale = np.random.uniform( 0., 1., size = num_sample )
    sample = np.hstack( [ x.reshape( -1, 1 ), y.reshape( -1, 1 ), scale.reshape( -1, 1 ) ] )
    label = np.zeros( [ num_sample, 2 ] )
    for i in range( num_sample ):
        if sample[ i, 0 ] * sample[ i, 1 ] > 0 and sample[ i, 2 ] >= thresh:
            label[ i, 1 ] = 1
        else:
            label[ i, 0 ] = 1
    

    # additional 
    num = num_sample // 4
    x = np.random.uniform( 0, 1, size = num )
    y = np.random.uniform( 0, 1, size = num )

    noise_x = np.random.normal(0., 0.2, size = num )
    noise_y = np.random.normal(0., 0.2, size = num )

    x += noise_x
    y += noise_y

    scale = np.random.uniform( 0.5, 1., size = num )
    additional_sample = np.hstack( [ x.reshape( -1, 1 ), y.reshape( -1, 1 ), scale.reshape( -1, 1 ) ] )
    additional_label = np.zeros( [ num, 2 ] )
    for i in range( num ):
        if additional_sample[ i, 0 ] * additional_sample[ i, 1 ] > 0 and additional_sample[ i, 2 ] >= thresh:
            additional_label[ i, 1 ] = 1
        else:
            additional_label[ i, 0 ] = 1

    sample = np.concatenate( [ sample, additional_sample ], axis = 0 )
    label = np.concatenate( [ label, additional_label ], axis = 0 )
    return sample, label

X, y = generate( num_sample = 2000 )
np.save( "new_xor_test_x.npy", X )
np.save( "new_xor_test_y.npy", y )


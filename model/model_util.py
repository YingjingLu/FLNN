import tensorflow as tf 

# stemming from pytorch hard tanh
def hard_tanh( inputs ):

    return tf.minimum( 1.0, tf.maximum( -1.0, inputs ) )


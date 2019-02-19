''' 
batch normalization
c: config
x: data
'''

def inference():
    c['conv_filters_out'] = 64
    c['ksize'] = 7
    c['stride'] = 2
    x = conv(x, c)
    x = bn(x, c)
    x = activation(x)

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001

def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    
    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias
    
    
    axis = list(range(len(x_shape) - 1))
    
    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)
    
    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)
    
    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    
    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))
    
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??
    
    return x

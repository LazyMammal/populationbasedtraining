from __future__ import print_function

import tensorflow as tf


def get_optimizer(optimizer='sgd'):
    loss_fn = tf.get_collection('loss_fn')[0]
    learning_rate = tf.get_collection('learning_rate')[0]

    if optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:  # 'sgd'
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    beta = 0.01
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    weights = tf.add_n([tf.nn.l2_loss(var) for var in var_list if var is not None])
    regularizer = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(loss_fn + beta * regularizer)
    train_step = opt.minimize(loss)

    opt_vars = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list if var is not None]
    if isinstance(opt, tf.train.AdamOptimizer):
        opt_vars.extend([opt._beta1_power, opt._beta2_power])

    reset_opt = tf.variables_initializer(opt_vars)
    init_op = tf.global_variables_initializer()

    return train_step, init_op, reset_opt

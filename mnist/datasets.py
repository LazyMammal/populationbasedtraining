from __future__ import print_function

from importlib import import_module
import numpy as np
import tensorflow as tf
import mnist
from timer import Timer


def main():
    # feed_dict()
    datasets('conv_dropout_model', 'softmax')


def feed_dict():
    dataset = mnist.get_dataset('fashion')
    x, y_, train_step, learning_rate, accuracy = mnist.gen_model('conv_dropout_model', 'softmax')

    with tf.Session() as sess:
        print("feed_dict")
        sess.run(tf.global_variables_initializer())
        datasize = len(dataset.train.labels) // 4
        for batch_size in list(range(100, 3000, 100)):  # [1, 2, 4, 8, 16, 32, 64] +
            epoch_time = Timer()
            iterations = datasize // batch_size
            for _ in range(iterations):
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            split = epoch_time.split()
            print('%d, %d, %3.1fs, %d/s' % (batch_size, iterations, split, datasize // split))


def datasets(model, loss):
    modelmodule = import_module(model)
    lossmodule = import_module(loss)
    learning_rate = tf.placeholder_with_default(tf.constant(0.01, dtype=tf.float32), shape=[])

    mnist_dataset = mnist.get_dataset('fashion')
    datasize = len(mnist_dataset.train.labels)

    features = mnist_dataset.train.images
    labels = mnist_dataset.train.labels
    assert features.shape[0] == labels.shape[0]

    batch_size = 1000

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.take(datasize)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_example, next_label = iterator.get_next()

    model = modelmodule.make_model(next_example, next_label)
    loss_fn = lossmodule.make_loss(model, next_label)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn)

    with tf.Session() as sess:
        print("datasets")
        sess.run(tf.global_variables_initializer())
        epoch = 0
        epoch_time = Timer()
        iterations = datasize // batch_size
        while True:
            sess.run(iterator.initializer)
            for _ in range(iterations):
                sess.run(train_step)
            epoch += 1
            split = epoch_time.split()
            print('%d, %d, %d, %3.1fs, %d/s' % (epoch, batch_size, iterations, split, datasize // split))


def datasets_batch_size(model, loss):
    """
        this is a bit of hack.  normally one would NOT duplicate the dataset so many times.
        crashes Python interpretor after about 10 loops
    """
    modelmodule = import_module(model)
    lossmodule = import_module(loss)
    learning_rate = tf.placeholder_with_default(tf.constant(0.01, dtype=tf.float32), shape=[])

    mnist_dataset = mnist.get_dataset('fashion')
    datasize = len(mnist_dataset.train.labels) // 4

    features = mnist_dataset.train.images
    labels = mnist_dataset.train.labels
    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.take(datasize)
    dataset2 = dataset.batch(1)

    #iterator = dataset.make_one_shot_iterator()
    iterator = tf.data.Iterator.from_structure(dataset2.output_types, dataset2.output_shapes)
    dataset2 = None
    #handle = tf.placeholder(tf.string, shape=[])
    #iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    next_example, next_label = iterator.get_next()

    model = modelmodule.make_model(next_example, next_label)
    loss_fn = lossmodule.make_loss(model, next_label)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn)

    with tf.Session() as sess:
        print("datasets")
        sess.run(tf.global_variables_initializer())
        for batch_size in list(range(100, 3000, 100)):  # [1, 2, 4, 8, 16, 32, 64] +
            epoch_time = Timer()
            iterations = datasize // batch_size
            batch_dataset = dataset.batch(batch_size)
            #batch_iterator = batch_dataset.make_initializable_iterator()
            #batch_handle = sess.run(batch_iterator.string_handle())
            #sess.run(batch_iterator.initializer)
            sess.run(iterator.make_initializer(batch_dataset))
            print('# %3.1fs' % epoch_time.split())
            for _ in range(iterations):
                #sess.run(train_step, feed_dict={handle: batch_handle})
                sess.run(train_step)
            split = epoch_time.split()
            print('%d, %d, %3.1fs, %d/s' % (batch_size, iterations, split, datasize // split))


if __name__ == '__main__':
    main()

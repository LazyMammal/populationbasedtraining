from __future__ import print_function

import argparse
from importlib import import_module
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mnist
from timer import Timer


MNIST = mnist.get_dataset('fashion')


def main(args):
    # augment()
    # feed_dict()
    # datasets(args.model, args.loss, args.batch_size)
    for batch_size in [1, 2, 4, 8, 16, 32, 64] + list(range(100, 3000, 100)):
        if batch_size >= args.batch_size:
            tf.reset_default_graph()
            datasets(args.model, args.loss, batch_size)


def augment():
    test_dataset = tf.data.Dataset.from_tensor_slices((MNIST.test.images, MNIST.test.labels))
    dataset = sample_pair_dataset()

    test_iterator = test_dataset.make_one_shot_iterator()
    test_example, test_label = test_iterator.get_next()

    iterator = dataset.make_one_shot_iterator()
    next_example, next_label = iterator.get_next()

    with tf.device("cpu:0"):
        with tf.Session() as sess:
            plotnum = 1
            for _ in range(16):
                features = sess.run(next_example)
                plt.subplot(4, 4, plotnum)
                plt.imshow(features)
                plotnum += 1
            plt.show()


def gen_mnist_pairs():
    while True:
        yield MNIST.train.next_batch(2)


def _distort_images(pair, labels):
    result1, label1 = _distort_image(pair[0], labels[0])
    result2, label2 = _distort_image(pair[1], labels[1])
    return [result1, result2], [label1, label2]


def _distort_image(features, labels):
    with tf.variable_scope('distortimage', reuse=tf.AUTO_REUSE):
        image = tf.reshape(features, (28, 28, 1))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.grayscale_to_rgb(image)
        pad = tf.random_uniform([2], 0, 18, tf.int32)
        image = tf.image.crop_to_bounding_box(image, pad[0], pad[1], 28 - pad[0], 28 - pad[1])
        image = tf.image.pad_to_bounding_box(image, pad[0], pad[1], 28, 28)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return tf.squeeze(image), labels


def _mix_images(features, labels):
    return (.7 * features[0] + .3 * features[1]), labels[0]


def sample_pair_dataset():
    dataset = tf.data.Dataset.from_generator(gen_mnist_pairs, (tf.float32, tf.float32))
    dataset = dataset.map(_distort_images, 4)
    dataset = dataset.map(_mix_images, 4)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(100)
    #dataset = dataset.batch(300)
    return dataset


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


def datasets(modelname, loss, batch_size=1000):
    modelmodule = import_module(modelname)
    lossmodule = import_module(loss)
    learning_rate = tf.placeholder_with_default(tf.constant(0.01, dtype=tf.float32), shape=[])

    datasize = len(MNIST.train.labels)

    features = MNIST.train.images
    labels = MNIST.train.labels
    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.take(datasize)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_example, next_label = iterator.get_next()

    model = modelmodule.make_model(next_example, next_label)
    loss_fn = lossmodule.make_loss(model, next_label)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn)

    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
        print("datasets: batch_size %d, model %s" % (batch_size, modelname))
        sess.run(tf.global_variables_initializer())
        epoch = 0
        epoch_time = Timer()
        iterations = datasize // batch_size
        #for _ in range(100):
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

    datasize = len(MNIST.train.labels) // 4

    features = MNIST.train.images
    labels = MNIST.train.labels
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
            # sess.run(batch_iterator.initializer)
            sess.run(iterator.make_initializer(batch_dataset))
            print('# %3.1fs' % epoch_time.split())
            for _ in range(iterations):
                #sess.run(train_step, feed_dict={handle: batch_handle})
                sess.run(train_step)
            split = epoch_time.split()
            print('%d, %d, %3.1fs, %d/s' % (batch_size, iterations, split, datasize // split))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default="conv_dropout_model", help="tensorflow model (conv dropout)")
    parser.add_argument('--loss', nargs='?', default="softmax", help="tensorflow loss")
    parser.add_argument('--batch_size', nargs='?', type=int, default=1000, help="batch size (1000)")
    main(parser.parse_args())

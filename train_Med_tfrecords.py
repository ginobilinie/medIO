"""Train and Eval the MNIST network.

This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See tensorflow/g3doc/how_tos/reading_data.md#reading-from-files
for context.

YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import print_function

import os.path
import time

import tensorflow.python.platform
import numpy
import tensorflow as tf


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('train_dir', 'data', 'Directory with the training data.')

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
            'dataHeight': tf.FixedLenFeature([], tf.int64),
            'dataWidth': tf.FixedLenFeature([], tf.int64),
            'dataDepth': tf.FixedLenFeature([], tf.int64),
            'labelHeight': tf.FixedLenFeature([], tf.int64),
            'labelWidth': tf.FixedLenFeature([], tf.int64),
            'labelDepth': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    print(repr(image))
    
    dataHeight = tf.cast(features['dataHeight'], tf.int32)
    print(repr(dataHeight))
    dataWidth = tf.cast(features['dataWidth'], tf.int32)
    dataDepth = tf.cast(features['dataDepth'], tf.int32)
    
    im_shape = tf.pack([dataHeight, dataWidth, dataDepth])
      
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    #   label = tf.cast(features['label'], tf.int32) 
    new_image = tf.reshape(image, im_shape)
    print(repr(new_image))
    
    new_image = tf.Print(new_image, [im_shape])
        
    #image.set_shpe(im_shape)
    
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #   new_image2 = tf.cast(new_image, tf.float32) * (1. / 255) - 0.5
    
    label = tf.decode_raw(features['label_raw'], tf.float32)
    print(repr(label))
    
    labelHeight = tf.cast(features['labelHeight'], tf.int32)
    print(repr(labelHeight))
    labelWidth = tf.cast(features['labelWidth'], tf.int32)
    labelDepth = tf.cast(features['labelDepth'], tf.int32)
    
    label_shape = tf.pack([labelHeight, labelWidth, labelDepth])
    new_label = tf.reshape(label, label_shape)
    print(repr(new_label))
    new_label = tf.Print(new_label, [label_shape])
    
    return new_image, label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels


def run_training():
  """Train MNIST for a number of steps."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    # Add to the Graph operations that train the model.
    train_op = tf.constant([1])

    # The op for initializing the variables.
    init_op = tf.initialize_all_variables()

    # Create a session for running operations in the Graph.
    ###This is very important!!!!, we can use with dialogue instead, 
    #and we cannot add additional TF operations within this session
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #tf.finalize() #make it read only, we cannot create new TF operations downward if we use tf.finalize()
    #however, if we create new TF operations, and if we donot use tf.finalize(), there will be no error and exception, but the 
    #graph begame bigger and bigger, and then the speed will be seriously affected.

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()

        # Run one step of the model.  The return values are
        # the activations from the `train_op` (which is
        # discarded) and the `loss` op.  To inspect the values
        # of your ops or variables, you may include them in
        # the list passed to sess.run() and the value tensors
        # will be returned in the tuple from the call.
        sess.run(train_op)
        
        loss_value = 0

        duration = time.time() - start_time

        # Print an overview fairly often.
        if step % 100 == 0:
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                     duration))
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
##### This is also very important 


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()

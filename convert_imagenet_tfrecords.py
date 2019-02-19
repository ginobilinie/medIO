"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import print_function

import os
import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import csv
from scipy import misc

IMAGE_SRC = '/home/ces/imagenet/ILSVRC2012_256'

tf.app.flags.DEFINE_string('set_dir', '/home/ces/caffe/data/ilsvrc12/',
                           'Directory where train file is stored')

tf.app.flags.DEFINE_string('directory', 'data',
                           'Directory to download data files and write the '
                           'converted result')
                           
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




output_fname = os.path.join(FLAGS.directory, 'train' + '.tfrecords')
print("output_fname: %s" % output_fname);

train_fname = os.path.join(FLAGS.set_dir, 'train.txt')
print("train_fname: %s" % train_fname);

writer = tf.python_io.TFRecordWriter(output_fname)

map_label_to_dir = {}
with open(train_fname, 'rb') as csv_file:
    reader = csv.reader(csv_file, delimiter=' ')
    for row in reader:
		path = os.path.dirname(row[0])
		label = int(row[1])
		map_label_to_dir[label] = path

		src_fname = os.path.join(IMAGE_SRC + '/train', row[0]);
		print("%s %d" % (src_fname, label))
		#print(row);

		im = misc.imread(src_fname)
		image_raw = im.tostring()
		rows = im.shape[0]
		cols = im.shape[1]
		
		if np.ndim(im) == 3:
			depth = im.shape[2]
		else:
			depth = 1

		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(rows),
			'width': _int64_feature(cols),
			'depth': _int64_feature(depth),
			'label': _int64_feature(label),
			'image_raw': _bytes_feature(image_raw)}))
		writer.write(example.SerializeToString())


if __name__ == '__main__':
  tf.app.run()


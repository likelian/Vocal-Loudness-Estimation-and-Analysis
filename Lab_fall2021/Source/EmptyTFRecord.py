import tensorflow as tf

#import numpy as np
#import IPython.display as display


filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write()

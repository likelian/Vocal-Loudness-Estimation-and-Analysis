#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def _parse_function(example_proto):
  features = {"data": tf.io.FixedLenFeature((), tf.string),
              "label": tf.io.FixedLenFeature((), tf.int64)}
  parsed_features = tf.io.parse_single_example(example_proto, features)
  data = tf.io.decode_raw(parsed_features['data'], tf.float32)
  return data, parsed_features["label"]

def load_tfrecords(srcfile):
    sess = tf.compat.v1.Session()

    dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
    dataset = dataset.map(_parse_function) # parse data into tensor
    dataset = dataset.repeat(2) # repeat for 2 epoches
    dataset = dataset.batch(5) # set batch_size = 5

    #iterator = dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_data = iterator.get_next()

    while True:
        try:
            data, label = sess.run(next_data)
            print(data)
            print(label)
        except tf.errors.OutOfRangeError:
            break

load_tfrecords(srcfile="/Users/likelian/Desktop/Lab/Lab_fall2021/TFRecord/amy_1.tfrecords")

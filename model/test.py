# -*- coding: utf-8 -*-
from enum import Enum
import tensorflow as tf


def decode_libsvm(line):
    # columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
    # features = dict(zip(CSV_COLUMNS, columns))
    # labels = features.pop(LABEL_COLUMN)
    print(line)
    sess = tf.Session()
    columns = tf.string_split([line], ' ')
    labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
    print("labels")
    print(sess.run(labels))
    splits = tf.string_split(columns.values[1:], ':')
    print("splits")
    print(sess.run(splits))
    print(sess.run(splits.values))
    print(sess.run(splits.dense_shape))
    id_vals = tf.reshape(splits.values, splits.dense_shape)
    print("id_vals")
    print(sess.run(id_vals))
    print("feat")
    feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)

    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
    print(sess.run(feat_ids))
    print(sess.run(feat_vals))
    # feat_ids = tf.reshape(feat_ids,shape=[-1,FLAGS.field_size])
    # for i in range(splits.dense_shape.eval()[0]):
    #    feat_ids.append(tf.string_to_number(splits.values[2*i], out_type=tf.int32))
    #    feat_vals.append(tf.string_to_number(splits.values[2*i+1]))
    # return tf.reshape(feat_ids,shape=[-1,field_size]), tf.reshape(feat_vals,shape=[-1,field_size]), labels
    return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

if __name__ == '__main__':
    dataset = tf.data.TextLineDataset(["..\\test\\test.txt"]).map(decode_libsvm)
    print(dataset)
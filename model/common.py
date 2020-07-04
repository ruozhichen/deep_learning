# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.logging as log
log.set_verbosity(log.INFO)  # 这样才能屏幕上输出info信息
import pandas as pd

import config
from config import ColumnType, ColumnTransform

def load_train_data(train_path, train_idx, frac=0.1, label='gender'):
    # train_idx: 训练数据id, 0~8, 每份数据占比10%，9用于valid
    #            train_idx < int(1.0 / frac - 1)
    # frac=0.1
    df = pd.read_csv(train_path)
    # df = df.sample(frac=0.05).reset_index(drop=True)
    batch_size = int(len(df) * frac)
    train_idx = max(0, min(train_idx, int(1.0 / frac - 1) - 1))
    train_start = train_idx * batch_size
    train_df = df[train_start:train_start + batch_size]
    test_start = int(1.0 / frac - 1) * batch_size
    test_df = df[test_start:]
    train_x = train_df[config.FEATURE_COLS]
    train_y = train_df[label]
    valid_x = test_df[config.FEATURE_COLS]
    valid_y = test_df[label]
    log.info("use training data between %d and %d with %d samples." % (train_start, train_start + batch_size, batch_size))
    return (train_x, train_y), (valid_x, valid_y)

def load_test_data(test_path, test_idx, frac=0.05):
    df = pd.read_csv(test_path)
    batch_size = int(len(df) * frac)
    test_start = test_idx * batch_size
    if test_idx < int(1.0 / frac - 1):
        test_df = df[test_start:test_start + batch_size]
    else:
        test_df = df[test_start:]
    test_x = test_df[config.FEATURE_COLS]
    test_y = None
    log.info(
        "use testing data between %d and %d with %d samples." % (test_start, test_start + batch_size, batch_size))
    return test_x, test_y

def train_input_fn(features, labels, batch_size=256, num_epochs=1):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # shuffle(): 将数据打乱，数值越大，混乱程度越大
    # repeat(): 数据集重复了指定次数
    dataset = dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)
    # return dataset
    iterator = dataset.make_one_shot_iterator()  # 数据取一次后就丢弃了
    feature, label = iterator.get_next()
    return feature, label

def eval_input_fn(features, labels, batch_size=256):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    # return dataset
    iterator = dataset.make_one_shot_iterator()  # 数据取一次后就丢弃了
    if labels is None:
        feature = iterator.get_next()
        return feature
    else:
        feature, label = iterator.get_next()
        return feature, label

def build_model_column():
    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim ** 0.25)) + 3))
    feature_conf_dic = config.feature_dict
    wide_columns = []
    deep_columns = []
    wide_dim = 0
    deep_dim = 0
    for feature, conf in feature_conf_dic.items():
        log.info("feature : %s" % (feature))
        f_type, f_tran, f_param, f_dtype, is_deep, is_wide = conf["type"], conf["transform"], conf["parameter"], \
                                                             conf['dtype'], conf['is_deep'], conf['is_wide']
        if f_type == ColumnType.CATEGORY:
            if f_tran == ColumnTransform.HASH_BUCKET:
                hash_bucket_size = f_param
                embed_dim = embedding_dim(hash_bucket_size)
                col = tf.feature_column.categorical_column_with_hash_bucket(feature,
                                                                            hash_bucket_size=hash_bucket_size,
                                                                            dtype=f_dtype)
                if is_wide:
                    wide_columns.append(col)
                    wide_dim += hash_bucket_size
                if is_deep:
                    deep_columns.append(tf.feature_column.embedding_column(col,
                                                                           dimension=embed_dim,
                                                                           combiner='sqrtn',
                                                                           initializer=tf.uniform_unit_scaling_initializer(),
                                                                           ckpt_to_load_from=None,
                                                                           tensor_name_in_ckpt=None,
                                                                           max_norm=None,
                                                                           trainable=True))

                    deep_dim += embed_dim
            elif f_tran == ColumnTransform.VOCAB:
                # len(vocab)+num_oov_buckets
                col = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                vocabulary_list=f_param,
                                                                                dtype=f_dtype,
                                                                                default_value=0,
                                                                                num_oov_buckets=0)
                if is_wide:
                    wide_columns.append(col)
                    wide_dim += len(f_param)
                if is_deep:
                    deep_columns.append(tf.feature_column.indicator_column(col))
                    deep_dim += len(f_param)
        elif f_type == ColumnType.CONTINUOUS:
            if f_tran == ColumnTransform.LOG:
                normalizer_fn = lambda x: tf.log(tf.cast(x, tf.float32) + tf.constant(1.0, tf.float32))
            else:
                normalizer_fn = None
            col = tf.feature_column.numeric_column(feature,
                                                   shape=(1,),
                                                   default_value=0,
                                                   # default None will fail if an example does not contain this column.
                                                   dtype=f_dtype,
                                                   normalizer_fn=normalizer_fn)
            if is_wide:
                wide_columns.append(col)
                wide_dim += 1
            if is_deep:
                deep_columns.append(col)
                deep_dim += 1
    log.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        log.debug('Wide columns: {}'.format(col))
    log.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        log.debug('Deep columns: {}'.format(col))
    log.info('Wide input dimension is: {}'.format(wide_dim))
    log.info('Deep input dimension is: {}'.format(deep_dim))
    return wide_columns, deep_columns
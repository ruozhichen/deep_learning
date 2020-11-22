# -*- coding: utf-8 -*-

import pandas as pd
import sys
import tensorflow as tf
import tensorflow.compat.v1.logging as log
log.set_verbosity(log.INFO)
import numpy as np
import shutil
import os
import json
import glob
from datetime import date, timedelta
from time import time
import random

sys.path.append("../preprocessing")
from config import ColumnType, ColumnTransform
from data_parser import FeatureDictionary, DataParser

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("label", 'gender', "col name of label")
tf.app.flags.DEFINE_integer("num_threads", 6, "Number of threads")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '..\\data\\sample', "data dir")
tf.app.flags.DEFINE_string("model_dir", '..\\data\\model', "model check point dir")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval}")
tf.app.flags.DEFINE_string("model_type", 'Inner', "model type {Inner, Outer}")


def input_fn(feat_ids, feat_vals, labels, batch_size=256, num_epochs=1, perform_shuffle=False):
    # 这里记得要将feat_vals和labels转化为tf.float32，否则在tf.multiply运算时会报错
    dataset = tf.data.Dataset.from_tensor_slices(({"feat_ids": tf.cast(feat_ids, tf.int32),
                                                   "feat_vals": tf.cast(feat_vals, tf.float32)},
                                                  tf.cast(labels, tf.float32)))
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    feature, label = iterator.get_next()
    return feature, label


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params["field_size"]  # 特征个数
    feature_size = params["feature_size"]  # 特征维度，即feature_dim
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    # ------bulid weights------
    Bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
    Linear_Weight = tf.get_variable(name='linear_weight', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    Feature_Embedding = tf.get_variable(name='feature_emb', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    # ------build feaure-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    num_pairs = int(field_size * (field_size - 1) / 2)
    # ------build f(x)------
    with tf.variable_scope("Linear-part"):
        feat_wgts = tf.nn.embedding_lookup(Linear_Weight, feat_ids)              # None * F
        # multiply为element-wise乘法
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), axis=1)  # None * 1

    with tf.variable_scope("Embedding-layer"):
        embeddings = tf.nn.embedding_lookup(Feature_Embedding, feat_ids)
        # reshape，保证其第一第二维度和embedding一致
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)  # None * F * M

    # 非优化版本，两两计算
    with tf.variable_scope("PNN-layer"):
        if FLAGS.model_type == 'Inner':
            row = []
            col = []
            for i in range(field_size - 1):
                for j in range(i + 1, field_size):
                    row.append(i)
                    col.append(j)
            # 根据row,col中的索引，从embedding中取向量
            p = tf.gather(embeddings, row, axis=1)  # None * num_pairs * M
            q = tf.gather(embeddings, col, axis=1)  # None * num_pairs * M
            inner = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])  # None * num_pairs
            deep_inputs = tf.concat([tf.reshape(embeddings, shape=[-1, field_size * embedding_size]), inner], 1)  # None * ( F * M + F*(F-1)/2 )
        elif FLAGS.model_type == 'Outer':
            row = []
            col = []
            for i in range(field_size - 1):
                for j in range(i + 1, field_size):
                    row.append(i)
                    col.append(j)
            p = tf.gather(embeddings, row, axis=1)
            q = tf.gather(embeddings, col, axis=1)
            # einsum('i,j->ij', p, q)  # output[i,j] = p[i]*q[j]
            # 可以理解为每两个特征向量，做矩阵乘法，[M, 1] X [1,M] = [M, M]
            outer = tf.reshape(tf.einsum('api,apj->apij', p, q),
                               [-1, num_pairs * embedding_size * embedding_size])  # None * (num_pairs * M * M)
            deep_inputs = tf.concat([tf.reshape(embeddings, shape=[-1, field_size * embedding_size]), outer], 1)  # None * ( F * M + num_pairs * M * M )

    # 简化版本
    # with tf.variable_scope("PNN-layer-2"):
    #     if FLAGS.model_type:
    #         inner = tf.square(tf.reduce_sum(embeddings, axis=1))  # (N, K)
    #         deep_inputs = tf.concat([tf.reshape(embeddings, shape=[-1, field_size * embedding_size]), inner],
    #                                 1)  # None * (F*K + K)
    #     else:
    #         emb_sum = tf.reduce_sum(embeddings, axis=1)  # (N, K)
    #         # (N,K,1) X (N,1,K) = (N,K,K)
    #         outer = tf.matmul(tf.expand_dims(emb_sum, axis=2), tf.expand_dims(emb_sum, axis=1)) # (N, K, K)
    #         outer = tf.reshape(outer, [-1, embedding_size * embedding_size])
    #         deep_inputs = tf.concat([tf.reshape(embeddings, shape=[-1, field_size * embedding_size]), outer], 1) # None * ( F * K + K * K )

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False

        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i],
                                                            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                            scope='mlp%d' % i)
            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])
                # deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)
        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.nn.relu,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   scope='deep_out')
        y_d = tf.reshape(y_deep, shape=[-1])

    with tf.variable_scope("PNN-out"):
        # tf.ones_like: 该操作返回一个具有和给定tensor相同形状（shape）和相同数据类型（dtype），但是所有的元素都被设置为1的tensor
        y_bias = Bias * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
        y = y_bias + y_linear + y_d
        pred = tf.sigmoid(y)
    # logits:深度代码中出现的logits，可以理解为未归一化的概率, 即神经网络的一层输出结果。该输出一般会再接一个softmax/sigmod输出最终的概率
    predictions = {"prob": pred, "logits": y}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    # ------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(Linear_Weight) + \
        l2_reg * tf.nn.l2_loss(Feature_Embedding)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.compat.v1.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  # tf.compat.v1.train.get_global_step

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def build_model_estimator(model_params):
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads}),
        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)
    return DeepFM


# def my_auc(labels, predictions):
#     return {'auc': tf.compat.v1.metrics.auc(labels, predictions['logistic'])}


def main(_):
    feat_dict = FeatureDictionary()
    print("feature_size: %d" % feat_dict.feature_size)
    print("field_size: %d" % feat_dict.field_size)
    print(feat_dict.col2feat_id.keys())
    dataparser = DataParser(feat_dict, FLAGS.label)
    train_ids, train_vals, train_labels = dataparser.parse(infile="%s\\train_sample.csv" % FLAGS.data_dir)
    print("len of train: %d" % len(train_ids))
    test_ids, test_vals, test_labels = dataparser.parse(infile="%s\\test_sample.csv" % FLAGS.data_dir)
    print("len of test: %d" % len(test_ids))

    # ------bulid Tasks------
    model_params = {
        "field_size": feat_dict.field_size,
        "feature_size": feat_dict.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }
    print(model_params)
    DeepFM = build_model_estimator(model_params)
    # DeepFM = tf.contrib.estimator.add_metrics(DeepFM, my_auc)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_ids, train_vals, train_labels,
                                                                      num_epochs=FLAGS.num_epochs,
                                                                      batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(test_ids, test_vals, test_labels,
                                                                    num_epochs=1,
                                                                    batch_size=FLAGS.batch_size),
                                          steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(DeepFM, train_spec, eval_spec)
        results = DeepFM.evaluate(
            input_fn=lambda: input_fn(test_ids, test_vals, test_labels, num_epochs=1, batch_size=FLAGS.batch_size))
        for key in results:
            log.info("%s : %s" % (key, results[key]))
    elif FLAGS.task_type == 'eval':
        results = DeepFM.evaluate(input_fn=lambda: input_fn(test_ids, test_vals, test_labels,
                                                            num_epochs=1, batch_size=FLAGS.batch_size))
        for key in results:
            log.info("%s : %s" % (key, results[key]))
    elif FLAGS.task_type == 'infer':
        preds = DeepFM.predict(input_fn=lambda: input_fn(test_ids, test_vals, test_labels,
                                                         num_epochs=1, batch_size=FLAGS.batch_size),
                               predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

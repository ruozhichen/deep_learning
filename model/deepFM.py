# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1.logging as log
log.set_verbosity(log.INFO)  # 这样才能屏幕上输出info信息

import numpy as np

from config import ColumnType, ColumnTransform
import config
import common

#import argparse
import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time
#import gc
#from multiprocessing import Process

#import math
import random
#import pandas as pd
#import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

def input_fn(feat_ids, feat_vals, labels, batch_size=256, num_epochs=1, perform_shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(({"feat_ids": feat_ids, "feat_vals": feat_vals}, labels))
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    feature, label = iterator.get_next()
    return feature, label

def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    #------hyperparameters----
    field_size = params["field_size"]  # 特征个数
    feature_size = params["feature_size"]  # 特征维度，即feature_dim
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    #batch_norm_decay = params["batch_norm_decay"]
    #optimizer = params["optimizer"]
    layers  = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))

    #------bulid weights------
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1, field_size])

    #------build f(x)------
    with tf.variable_scope("First-order"):
        feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)              # None * F
        # multiply为element-wise乘法
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), axis=1)  # None * 1

    with tf.variable_scope("Second-order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)  # None * F * K
        # reshape，保证其第一第二维度和embedding一致
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)  # None * F * K
        sum_square = tf.square(tf.reduce_sum(embeddings,1))  # None * K
        square_sum = tf.reduce_sum(tf.square(embeddings),1)  # None * K
        y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)	# None * 1

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            #normalizer_fn = tf.contrib.layers.batch_norm
            #normalizer_fn = tf.layers.batch_normalization
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': True, 'reuse': None}
            else:
                train_phase = False
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': False, 'reuse': True}
        else:
            normalizer_fn = None
            normalizer_params = None

        deep_inputs = tf.reshape(embeddings,shape=[-1,field_size*embedding_size]) # None * (F*K)
        for i in range(len(layers)):
            #if FLAGS.batch_norm:
            #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
                #normalizer_params.update({'scope': 'bn_%d' %i})
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                #normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                #deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')
        y_d = tf.reshape(y_deep,shape=[-1])
        #sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]], initializer=tf.glorot_normal_initializer())
        #sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        #deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

    with tf.variable_scope("DeepFM-out"):
        #y_bias = FM_B * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
        y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)      # None * 1
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(FM_W) + \
        l2_reg * tf.nn.l2_loss(FM_V)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

def build_model_estimator(num_class):
    pass

def my_auc(labels, predictions):
    return {'auc': tf.compat.v1.metrics.auc(labels, predictions['logistic'])}


def main(idx):
    pass



if __name__ == '__main__':
    # i=14和18的时候，数据出了问题
    # Unable to allocate 2.47 GiB for an array with shape (11, 30082771) and data type int64
    # for i in range(15, 19):
    #     print("****************** %d ********************" % i)
    #     main(i)
    if FLAGS.task_type != 'infer':
        upper = int(1 / FLAGS.frac) - 1
    else:
        upper = int(1 / FLAGS.frac)
    # upper = 1
    for i in range(0, 1):
        print("****************** %s: %d ********************" % (FLAGS.task_type, i))
        main(i)
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
tf.app.flags.DEFINE_integer("tasks_num", 2, "the number of tasks")


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
    experts_num = params['experts_num']
    experts_units = params['experts_units']
    use_experts_bias = params['use_experts_bias']
    use_gate_bias = params['use_gate_bias']


    # ------bulid weights------
    feature_embedding = tf.get_variable(name='feature_emb', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())
    # ------build feaure-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])
    # ------build f(x)------
    with tf.variable_scope("Embedding-layer"):
        embeddings = tf.nn.embedding_lookup(feature_embedding, feat_ids)
        # reshape，保证其第一第二维度和embedding一致
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)  # None * F * M
        input_layer = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])
    # experts
    # feature_dim * experts_units * experts_num
    experts_weight = tf.get_variable(name='experts_weight',
                                     dtype=tf.float32,
                                     shape=(input_layer.get_shape()[1], experts_units, experts_num),
                                     initializer=tf.contrib.layers.xavier_initializer())
    experts_bias = tf.get_variable(name='expert_bias',
                                   dtype=tf.float32,
                                   shape=(experts_units, experts_num),
                                   initializer=tf.contrib.layers.xavier_initializer())

    # gates
    # tasks_num * feature_dim * experts_num
    gate_weights = [tf.get_variable(name='gate%d_weight' % i,
                                   dtype=tf.float32,
                                   shape=(input_layer.get_shape()[1], experts_num),
                                   initializer=tf.contrib.layers.xavier_initializer())
                    for i in range(FLAGS.tasks_num)]
    # tasks_num * experts_num
    gate_biases = [tf.get_variable(name='gate%d_bias' % i,
                                 dtype=tf.float32,
                                 shape=(experts_num,),
                                 initializer=tf.contrib.layers.xavier_initializer())
                   for i in range(FLAGS.tasks_num)]

    with tf.variable_scope("MMOE-part"):
        # N * experts_units * experts_num
        experts_output = tf.tensordot(a=input_layer, b=experts_weight, axes=1) # axes=k 表示取a的后k维跟b的前k维进行矩阵相乘
        if use_experts_bias:
            experts_output = tf.add(experts_output, experts_bias)
        gates_output = []
        for i in range(FLAGS.tasks_num):
            # N * experts_num
            res = tf.matmul(input_layer, gate_weights[i])
            if use_gate_bias:
                res = tf.add(res, gate_biases[i])
            gates_output.append(res)
        # tasks_num * N * experts_num
        gate_outputs = tf.nn.softmax(gates_output)
        final_outputs = []
        for i in range(FLAGS.tasks_num):
            # N * 1 * experts_num
            expanded_gate_output = tf.expand_dims(gate_outputs[i], axis=1)
            # N * experts_units * experts_num
            weighted_expert_output = tf.multiply(experts_output, expanded_gate_output)
            # N * experts_units
            task_output = tf.reduce_sum(weighted_expert_output, axis=2)
            final_outputs.append(task_output)
        # 本数据不支持多任务学习，故这里将多任务输出的结果进行拼接后，作为DNN的输入。
        deep_inputs = tf.concat(final_outputs, axis=1)  # N * ( 2 * experts_unit)


    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False

        for i in range(1, len(layers)):
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

    with tf.variable_scope("MMOE-out"):
        y_d = tf.reshape(y_deep, shape=[-1])
        pred = tf.sigmoid(y_d)
    # logits:深度代码中出现的logits，可以理解为未归一化的概率, 即神经网络的一层输出结果。该输出一般会再接一个softmax/sigmod输出最终的概率
    predictions = {"prob": pred, "logits": y_d}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    # ------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(feature_embedding)

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
        "dropout": FLAGS.dropout,
        "experts_num": 3,
        "experts_units": 32,
        "use_experts_bias": True,
        "use_gate_bias": True
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

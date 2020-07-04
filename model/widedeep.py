# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1.logging as log
log.set_verbosity(log.INFO)  # 这样才能屏幕上输出info信息

import numpy as np

import config
import common


FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
# tf.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
# tf.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# tf.flags.DEFINE_integer("num_threads", 16, "Number of threads")
# tf.flags.DEFINE_integer("feature_size", 0, "Number of features")
# tf.flags.DEFINE_integer("field_size", 0, "Number of fields")
# tf.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
tf.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
# tf.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
# tf.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
# tf.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
# tf.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.flags.DEFINE_string("data_dir", 'd:\\Tencent\\code\\data', "data dir")
# tf.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.flags.DEFINE_string("model_dir", 'd:\\Tencent\\code\\data\\model_wd', "model check point dir")
# tf.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.flags.DEFINE_string("label", 'gender', "label type {gender, age}")
tf.flags.DEFINE_integer("train_idx", 0, "use ith training data")
tf.flags.DEFINE_integer("test_idx", 0, "use ith testing data")
tf.flags.DEFINE_integer("train_step", 1000, "training steps")
tf.flags.DEFINE_integer("num_calss", 2, "training steps")
tf.flags.DEFINE_float("frac", 0.1, "fraction of data each training")
# tf.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


def build_model_estimator(num_class):
    wide_columns, deep_columns = common.build_model_column()
    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=FLAGS.model_dir + "_%s" % FLAGS.label,
        n_classes=num_class,
        # wide settings
        linear_feature_columns=wide_columns,
        linear_optimizer=tf.compat.v1.train.FtrlOptimizer(
            learning_rate=0.01,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.001),
        # deep settings
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[256,128,64],
        dnn_optimizer=tf.compat.v1.train.AdagradOptimizer(
            learning_rate=0.005,
            initial_accumulator_value=0.1,
            use_locking=False))
    return model

def my_auc(labels, predictions):
    return {'auc': tf.compat.v1.metrics.auc(labels, predictions['logistic'])}


def main(idx):
    #WARNING:tensorflow:From deepFM.py:141: The name tf.logging.debug is deprecated. Please use tf.compat.v1.logging.debug instead.
    #WARNING:tensorflow:From deepFM.py:156: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
    model = build_model_estimator(FLAGS.num_calss)
    model = tf.contrib.estimator.add_metrics(model, my_auc)

    if FLAGS.task_type == 'train':
        log.info("Begin to train.")
        (train_x, train_y), (eval_x, eval_y) = common.load_train_data(FLAGS.data_dir + "\\train\\data_user.csv", idx, frac=FLAGS.frac, label=FLAGS.label)
        print(train_x.shape, train_y.shape)
        print(eval_x.shape, eval_y.shape)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: common.train_input_fn(train_x, train_y, FLAGS.batch_size, FLAGS.num_epochs),max_steps=FLAGS.train_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: common.eval_input_fn(eval_x, eval_y), steps=100,
                                          start_delay_secs=120, throttle_secs=600)
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
        results = model.evaluate(input_fn=lambda: common.eval_input_fn(eval_x, eval_y))
        for key in results:
            log.info("%s : %s" % (key, results[key]))
        # with tf.gfile.Open('%s\metrics\\train_auc' % FLAGS.model_dir, 'w') as f:
        #     f.write("auc %s" % results['auc'])
    elif FLAGS.task_type == 'eval':
        (train_x, train_y), (eval_x, eval_y) = common.load_train_data(FLAGS.data_dir + "\\train\\data_user.csv", idx, frac=FLAGS.frac, label=FLAGS.label)
        print(train_x.shape, train_y.shape)
        print(eval_x.shape, eval_y.shape)
        # results = model.evaluate(input_fn=lambda: common.eval_input_fn(eval_x, eval_y))
        results = model.evaluate(input_fn=lambda: common.eval_input_fn(train_x, train_y))
        for key in results:
            log.info("%s : %s" % (key, results[key]))
        # log.info('\nTest set accuracy: {accuracy:0.3f}\n'.format(**results))
    elif FLAGS.task_type == 'infer':
        test_x, test_y = common.load_test_data(FLAGS.data_dir+"\\test\\data_user.csv", idx, FLAGS.frac)
        pred_result = model.predict(input_fn=lambda: common.eval_input_fn(test_x, test_y))
        with open(FLAGS.data_dir+"\\res_age\\%s_res_%d.txt" % (FLAGS.label, idx), "w") as fout:
            for prob in pred_result:
                #  Expected to run at least one output from dict_keys(['logits', 'logistic', 'probabilities', 'class_ids', 'classes', 'all_class_ids', 'all_classes']), provided prob.
                fout.write("%s\n" % str(prob['classes']))
                # fout.write("%f\n" % (prob['classes']))

    # tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    # log.info('=' * 60)
    # FLAGS.model_dir = FLAGS.model_dir + FLAGS.biz_date
    # if FLAGS.job_name == 'chief':
    #     results = model.evaluate(input_fn=eval_input_fn)
    #     for key in results:
    #         log.info("%s : %s" % (key, results[key]))
    #     with tf.gfile.Open('%s/metrics/train_auc' % FLAGS.model_dir, 'w') as f:
    #         f.write("auc %s" % results['auc'])
    #     if FLAGS.incremental == 'True':
    #         model.export_savedmodel(os.path.join(FLAGS.model_dir, "incremental_model"), raw_serving_input_receiver_tensor_fn())
    #     else:
    #         model.export_savedmodel(os.path.join(FLAGS.model_dir, "model"), raw_serving_input_receiver_tensor_fn())
    #     cache = tf.summary.FileWriterCache()
    #     cache.clear()

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
    for i in range(0, upper):
        print("****************** %s: %d ********************" % (FLAGS.task_type, i))
        main(i)
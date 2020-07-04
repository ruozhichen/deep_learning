# -*- coding: utf-8 -*-
from enum import Enum
import tensorflow as tf

if __name__ == '__main__':
    a = [[[1, 2], [3, 4], [5, 6]],
         [[7, 8], [9, 10], [11, 12]],
         [[13, 14], [15, 16], [17, 18]],
         [[19, 20], [21, 22], [23, 24]]]

    b = [[[1], [2], [3]],
         [[4], [5], [6]],
         [[7], [8], [9]],
         [[10], [11], [12]]]

    c = tf.multiply(a, b)
    d = tf.reduce_sum(a,2)
    with tf.Session() as sess:
        print(sess.run(c))
        print("*******************")
        print(sess.run(d))
    # 4,3,1
    # a = [[[1], [2], [3]],
    #      [[4], [5], [6]],
    #      [[7], [8], [9]],
    #      [[10], [11], [12]]]
    # b = [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9],
    #      [10, 11, 12]]
    # c = tf.multiply(a, b)
    # with tf.Session() as sess:
    #     print(sess.run(c))
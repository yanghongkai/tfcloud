#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

v=tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名将原来变量v的滑动平均值直接赋值给v
saver=tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "model/model2.ckpt")
    print(sess.run(v))
















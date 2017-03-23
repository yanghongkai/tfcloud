#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

v=tf.Variable(0, dtype=tf.float32, name="v")
ema=tf.train.ExponentialMovingAverage(0.99)
maintain_average_op=ema.apply([v])
init_op=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    # shadow 初始值
    print(sess.run([v,ema.average(v)]))
    sess.run(tf.assign(v,10))
    # 为执行update操作
    print(sess.run([v,ema.average(v)]))
    # 执行op操作
    sess.run(maintain_average_op)
    print(sess.run([v,ema.average(v)]))







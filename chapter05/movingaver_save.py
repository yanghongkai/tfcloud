#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

v=tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.all_variables():
    print(variables.name)

ema=tf.train.ExponentialMovingAverage(0.99)
maintain_average_op=ema.apply(tf.all_variables())
for variables in tf.all_variables():
    print(variables.name)

init_op=tf.initialize_all_variables()
saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)
    # 保存的时候会将 v:0 v/ExponentialMovingAverage:0 这两个变量都保存下来
    saver.save(sess, "model/model2.ckpt")
    print(sess.run([v, ema.average(v)]))








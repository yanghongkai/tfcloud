#!/usr/bin/env python
# coding=utf-8

from __future__ import print_funciton
import tensorflow as tf

v1=tf.Variable(tf.constant(1.0, shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0, shape=[1]),name="v2")
result=v1+v2

init_op=tf.initialize_all_variables()
# declare tf.train.Saver used to save model
saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # save model to model/model.ckpt file
    saver.save(sess, "model/model.ckpt")





#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

# 这里声明的变量名称和已经保存的模型中变量的名称不同
v1=tf.Variable(tf.constant(1.0, shape=[1]),name="other-v1")
v2=tf.Variable(tf.constant(2.0, shape=[1]),name="other-v2")
result=v1+v2

# 使用一个字典dict 来重命名变量就可以加载原来的模型了。这个字典自定了原来名称为v1的变量现在加载到变量v1中（名称为other-v1)，名称为
# v2的变量加载到变量v2中（名称为other-v2).
saver=tf.train.Saver({"v1": v1, "v2": v2})

with tf.Session() as sess:
    saver.restore(sess,"model/model.ckpt")
    print(sess.run(result))









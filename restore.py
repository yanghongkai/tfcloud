#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

# 使用和保存模型代码中一样的方式来声明变量
v1=tf.Variable(tf.constant(1.0, shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0, shape=[1]),name="v2")
result=v1+v2

saver=tf.train.Saver()
with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess,"model/model.ckpt")
    print(sess.run(result))






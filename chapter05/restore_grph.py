#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

# 直接加载持久化的图
saver=tf.train.import_meta_graph("model/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess,"model/model.ckpt")
    # 通过张量名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))









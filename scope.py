#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

v1=tf.get_variable("v",[1])
print(v1.name)
# v:0 "v"为变量的名称，":0"表示这个变量是生成变量这个运算的第一个结果
with tf.variable_scope("foo"):
    v2=tf.get_variable("v",[1])
    print(v2.name) # foo/v:0
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3=tf.get_variable("v",[1])
        print(v3.name) # foo/bar/v:0

    v4=tf.get_variable("v1",[1])
    print(v4.name) # foo/v1:0

# 创建一个名称为空的命名空间，并设置reuse=True
with tf.variable_scope("foo",reuse=True):
    v5=tf.get_variable("bar/v",[1])
    print(v5.name)
    # 可以直接通过命名空间名称的变量名来获取其他命名空间下的变量。比如这里通过指定名称foo/bar/v来获取在命名空间foo/bar中创建的变量
    print(v5==v3)
    v6=tf.get_variable("v1",[1])
    print(v6==v4)
    print(v6.name)













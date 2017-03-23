#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf

v=tf.Variable(0, dtype=tf.float32, name="v")
ema=tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())















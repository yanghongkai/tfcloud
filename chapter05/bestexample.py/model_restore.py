#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import time 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path','mnist-data', 'data dir')
tf.app.flags.DEFINE_float('learning_rate_base',0.8, 'learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay',0.99, 'learning rate decay')
tf.app.flags.DEFINE_float('moving_average_decay',0.99, 'moving average decay')
tf.app.flags.DEFINE_float('regularization_rate',0.001, 'regularization rate')
tf.app.flags.DEFINE_integer('training_steps',30000, 'traing steps')
tf.app.flags.DEFINE_integer('n_hidden',500, 'hidden unit number')
tf.app.flags.DEFINE_integer('n_classes',10, 'label')
tf.app.flags.DEFINE_integer('n_input',784, 'input')
tf.app.flags.DEFINE_integer('batch_size',100, 'input')




class Model:
    def __init__(self,trainMode=True,reuse=False):
        self.n_classes=FLAGS.n_classes
        self.regularizer=None
        self.reuse=reuse
        self.train_op=None
        self.X=tf.placeholder(tf.float32,[None,FLAGS.n_input],name='x-input')
        self.Y=tf.placeholder(tf.float32,[None,FLAGS.n_classes],name='y-input')
        #self.learning_rate=FLAGS.learning_rate_base
        self.global_step=tf.Variable(0,trainable=False)
        self.learning_rate=tf.train.exponential_decay(FLAGS.learning_rate_base,self.global_step, 1000,FLAGS.learning_rate_decay, staircase=True)
        if trainMode:
            self.regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularization_rate)

    def get_weight_variable(self,shape, regularizer):
        weights=tf.get_variable("weights",shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def inference(self,X,regularizer,reuse):
        with tf.variable_scope('layer1',reuse=reuse):
            weights=self.get_weight_variable([FLAGS.n_input,FLAGS.n_hidden],self.regularizer)
            biases=tf.get_variable("biases",[FLAGS.n_hidden],initializer=tf.constant_initializer(0.0))
            layer1=tf.nn.relu(tf.matmul(X,weights)+biases)
        with tf.variable_scope('layer2',reuse=reuse):
            weights=self.get_weight_variable([FLAGS.n_hidden,FLAGS.n_classes],self.regularizer)
            biases=tf.get_variable("biases",[FLAGS.n_classes], initializer=tf.constant_initializer(0.0))
            layer2=tf.matmul(layer1,weights)+biases
        return layer2

    def loss(self):
        logits=self.inference(self.X,self.regularizer,self.reuse)
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits,tf.argmax(self.Y,1))
        cross_entropy_mean=tf.reduce_mean(cross_entropy)
        loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
        self.loss=loss
        correct_prediction=tf.equal(tf.argmax(logits,1), tf.argmax(self.Y,1))
        self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        self.train_op=tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,global_step=self.global_step)

    

def main(argv=None):
    mnist=input_data.read_data_sets(FLAGS.data_path,one_hot=True)
    variable_averages=tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
    variables_to_restore=variable_averages.variables_to_restore()
    saver=tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        #m=Model(trainMode=True)
        mvalid=Model(trainMode=False,reuse=True)
        #mtest=Model(trainMode=False,reuse=True)

        #m.loss()
        mvalid.loss()
        #init_op=tf.initialize_all_variables()
        #sess.run(init_op)
        #m.loss()
        xs,ys=mnist.train.next_batch(FLAGS.batch_size)
        ckpt=tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # reload model
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            acc=sess.run(mvalid.accuracy,feed_dict={mvalid.X:xs, mvalid.Y:ys})
            print("Epchs %d, Acc:%g" % (global_step,acc))


if __name__=='__main__':
    main()



































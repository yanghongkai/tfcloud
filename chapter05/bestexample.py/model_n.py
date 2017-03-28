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
    def __init__(self,trainMode=True):
        self.n_classes=FLAGS.n_classes
        self.regularizer=None
        self.train_op=None
        self.X=tf.placeholder(tf.float32,[None,FLAGS.n_input],name='x-input')
        self.Y=tf.placeholder(tf.float32,[None,FLAGS.n_classes],name='y-input')
        if trainMode:
            self.regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularization_rate)
        #with tf.variable_scope('layer1'):
        #    self.weights_ly1=self.get_weight_variable([FLAGS.n_input,FLAGS.n_hidden], self.regularizer)
        #    self.biases_ly1=tf.get_variable("biases",[FLAGS.n_hidden], initializer=tf.constant_initializer(0.0))
        #with tf.variable_scope('layer2'):
        #    self.weights_ly2=self.get_weight_variable([FLAGS.n_hidden,FLAGS.n_classes],self.regularizer)
        #    self.biases_ly2=tf.get_variable("biases",[FLAGS.n_classes],initializer=tf.constant_initializer(0.0))
        
        with tf.variable_scope('layer1'):
            self.weights_ly1=tf.get_variable(shape=[FLAGS.n_input,FLAGS.n_hidden],initializer=tf.truncated_normal_initializer(stddev=0.01), name="weights")
            self.biases_ly1=tf.get_variable("biases",shape=[FLAGS.n_hidden], initializer=tf.constant_initializer(0.0))
        with tf.variable_scope('layer2'):
            self.weights_ly2=tf.get_variable(shape=[FLAGS.n_hidden,FLAGS.n_classes], initializer=tf.truncated_normal_initializer(stddev=0.01),name="weights")
            self.biases_ly2=tf.get_variable("biases",shape=[FLAGS.n_classes],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(self.X,self.weights_ly1)+self.biases_ly1)
        layer2=tf.matmul(layer1,self.weights_ly2)+self.biases_ly2
        #logits=self.inference(self.X,self.regularizer)
        logits=layer2
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits,tf.argmax(self.Y,1))
        cross_entropy_mean=tf.reduce_mean(cross_entropy)
        loss=cross_entropy_mean
        #loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
        self.loss=loss
        correct_prediction=tf.equal(tf.argmax(logits,1), tf.argmax(self.Y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        self.train_op=tf.train.GradientDescentOptimizer(FLAGS.learning_rate_base).minimize(loss)
        #print(self.train_op)
        

    def get_weight_variable(self,shape, regularizer):
        weights=tf.get_variable("weights",shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights


    
def train(total_loss):
    return tf.train.GradientDescentOptimizer(FLAGS.learning_rate_base).minimize(total_loss)

def run_epoch(sess,model,tX,tY):
    start_time=time.time()
    #train_op,loss_val, accuracy_val=model.loss(model.x,model.y)
    #_, loss_val, accuracy_val=sess.run([train_op,loss_val,accuracy_val],feed_dict={model.x:tX, model.y:tY})
    #print("accuracy %g", % (loss_val))
    


def main(argv=None):
    mnist=input_data.read_data_sets(FLAGS.data_path,one_hot=True)
    with tf.Session() as sess:
        initializer=tf.random_uniform_initializer(-0.1,0.1)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            m=Model(trainMode=True)
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            mvalid=Model(trainMode=False)
            mtest=Model(trainMode=False)

        init_op=tf.initialize_all_variables()
        sess.run(init_op)
        print("----------train_op------------")
        #print(train_op)
        print(m.train_op)
        #for i in range(FLAGS.training_steps):
        for i in range(10):
            xs,ys=mnist.train.next_batch(FLAGS.batch_size)
            print("hello--before")
            #print(m.X)
            #print(m.loss)
            #print(m.train_op)
            #feed_dict={}
            #feed_dict[m.X]=xs
            #feed_dict[m.Y]=ys
            #_, loss_val=sess.run([m.train_op,m.loss],feed_dict=feed_dict)
            _, loss_val=sess.run([m.train_op,m.loss],feed_dict={m.X:xs, m.Y:ys})
            print("Epochs %d, Loss:%g" % (i,loss_val))


if __name__=='__main__':
    main()




































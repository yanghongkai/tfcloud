#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置输入和输出节点个数配置神经网络的参数
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAING_STEPs=30000
MOVING_AVERAGE_DECAY=0.99

# 在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biase1, weights2, biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor, weights1)+biase1)
        return tf.matmul(layer1, weights2)+biases2
    else:
        layer1=tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1))+avg_class.average(biase1))
        return tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
    x=tf.placeholder(tf.float32, [None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32, [None,OUTPUT_NODE],name='y-input')
    weights1=tf.Variable(
        tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biase1=tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2=tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None
    y=inference(x, None, weights1, biase1, weights2, biases2)

    global_step=tf.Variable(0, trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    average_y=inference(x, variable_averages, weights1, biase1, weights2, biases2)

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(weights1)+regularizer(weights2)
    loss=cross_entropy_mean+regularization

    # 设置指数衰减的学习率
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op=tf.no_op(name='train')
    
    correct_prediction=tf.equal(tf.argmax(average_y,1), tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op=tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        validate_feed={x:mnist.validation.images, y_:mnist.validation.labels}
        test_feed={x:mnist.test.images, y_:mnist.test.labels}

        # 迭代地训练神经网络
        for i in range(TRAING_STEPs):
            if i%1000==0:
                validate_acc=sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy" "using average model is %g" % (i,validate_acc))
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_:ys})

        test_acc=sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average" "model is %g" % (TRAING_STEPs, test_acc))

def main(argv=None):
    mnist=input_data.read_data_sets("mnist-data",one_hot=True)
    train(mnist)

if __name__ =='__main__':
    tf.app.run()





















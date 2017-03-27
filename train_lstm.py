#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print(sys.getdefaultencoding())


import numpy as np
from sklearn import metrics

import tensorflow as tf
from tensorflow.contrib import learn
import os
import yfile

FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path',"/Users/tech/code/kcws/train.txt",'Training data dir')
tf.app.flags.DEFINE_string('test_data_path',"./test.txt",'Test data dir')
tf.app.flags.DEFINE_string('log_dir',"logs",'The log dir')
tf.app.flags.DEFINE_string('word2vec_path',"./vec.txt",'the word2vec data path')
tf.app.flags.DEFINE_integer('max_sentence_len',80,'max num of tokens per query')
tf.app.flags.DEFINE_integer('embedding_size',50,"embedding size")
tf.app.flags.DEFINE_integer('num_tags',16,'BMEO')
tf.app.flags.DEFINE_integer('num_hidden',100,'hidden unit number')
tf.app.flags.DEFINE_integer('batch_size',100,'num example per mini batch')
tf.app.flags.DEFINE_integer('train_steps',50000,'training steps')
tf.app.flags.DEFINE_float("learning_rate",0.002,'learning rate')


def do_load_data(path):
    x=[]
    y=[]
    fp=open(path,"r")
    for line in fp.readlines():
        line=line.rstrip()
        if not line:
            continue
        ss=line.split(" ")
        assert(len(ss)==(FLAGS.max_sentence_len*2))
        lx=[]
        ly=[]
        for i in xrange(FLAGS.max_sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i+FLAGS.max_sentence_len]))
        x.append(lx)
        y.append(ly)
    fp.close()
    return np.array(x),np.array(y)



class Model:
    def __init__(self,embeddingSize,distinctTagNum,c2vPath,numHidden):
        self.embeddingSize=embeddingSize
        self.distinctTagNum=distinctTagNum
        self.numHidden=numHidden
        self.c2v=self.load_w2v(c2vPath)
        print(len(self.c2v))
        # len(self.c2v)=13544 file:vectrain_50.txt 13542 50
        # c2v[0]=[0,0,0,0...] 表示未填充的idx  c2v[lenghth+1]存储平均，用于表示unk
        #print(self.c2v[0])
        #print(self.c2v[-1])
        self.words=tf.Variable(self.c2v,name="words")
        with tf.variable_scope('Softmax') as scope:
            self.W=tf.get_variable(shape=[numHidden*2,distinctTagNum],initializer=tf.truncated_normal_initializer(stddev=0.01),name="weights",
                                   regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b=tf.Variable(tf.zeros([distinctTagNum],name="bias"))
        self.trains_params=None
        self.inp=tf.placeholder(tf.int32,shape=[None,FLAGS.max_sentence_len],name="input_placeholder")
        pass

    def length(self,data):
        used=tf.sign(tf.reduce_max(tf.abs(data),reduction_indices=2))
        # data shape [100,80,50] reduce_max shape [100,80] used shape [100,80]
        length=tf.reduce_sum(used,reduction_indices=1)
        # length shape [100]
        length=tf.cast(length,tf.int32)
        return length

    def inference(self,X,reuse=None,trainMode=True):
        word_vectors=tf.nn.embedding_lookup(self.words,X)
        # word_vectors shape [100,80,50]
        length=self.length(word_vectors)
        # length shape [100]
        length_64=tf.cast(length,tf.int64)
        if trainMode:
            word_vectors=tf.nn.dropout(word_vectors,0.5)
        with tf.variable_scope("rnn_fwbw",reuse=reuse) as scope:
            forward_output,_=tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.numHidden),
                word_vectors,dtype=tf.float32,sequence_length=length,scope="RNN_forward")
            # forward_output shape [batch_size,time_steps,num_hidden] [100,80,100]
            backward_output_,_=tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(word_vectors,length_64,seq_dim=1),
                dtype=tf.float32,sequence_length=length,scope="RNN_backward")
            # backward_output_ shape [100,80,100]
            backward_output=tf.reverse_sequence(backward_output_,length_64,seq_dim=1)
            output=tf.concat(2,[forward_output,backward_output])
            # output shape [100,80,200]
            output=tf.reshape(output,[-1,self.numHidden*2])
            # output [100*80,200]
            if trainMode:
                output=tf.nn.dropout(output,0.5)
            matricized_unary_scores=tf.matmul(output,self.W)+self.b
            unary_scores=tf.reshape(matricized_unary_scores,[-1,FLAGS.max_sentence_len,self.distinctTagNum])
            # unary_scores shape [100,80,4]
            return unary_scores,length

    def loss(self,X,Y):
        P,sequence_length=self.inference(X)
        # crf input shape [batch_size,max_seq_len,num_tags] shape [100,80,4]
        log_likelihood,self.transition_params=tf.contrib.crf.crf_log_likelihood(P,Y,sequence_length)
        # log_likelihood shape [100]
        loss=tf.reduce_mean(-log_likelihood)
        return loss,log_likelihood

    def test_unary_score(self):
        P,sequence_length=self.inference(self.inp,reuse=True,trainMode=False)
        return P,sequence_length





    def load_w2v(self,path):
        fp=open(path,"r")
        print("load data from:",path)
        line=fp.readline().strip()
        ss=line.split(" ")
        total=int(ss[0])
        dim=int(ss[1])
        assert(dim==(FLAGS.embedding_size))
        ws=[]
        mv=[0 for i in range(dim)]
        # The first for 0
        ws.append([0 for i in range(dim)])
        for t in range(total):
            line=fp.readline().strip()
            ss=line.split(" ")
            assert(len(ss)==(dim+1))
            vals=[]
            for i in range(1,dim+1):
                fv=float(ss[i])
                mv[i-1]+=fv
                vals.append(fv)
            ws.append(vals)
        for i in range(dim):
            mv[i]=mv[i]/total
        ws.append(mv)
        fp.close()
        return np.asarray(ws,dtype=np.float32)


def read_csv(batch_size,file_name):
    filename_queue=tf.train.string_input_producer(file_name)
    #print(filename_queue)
    # FIFQQueue
    reader=tf.TextLineReader(skip_header_lines=0)
    key,value=reader.read(filename_queue)
    #print('-----------key,value---------')
    #print(key)
    #print(value)
    # key: Tensor("ReaderRead:0", shape=(), dtype=string)
    # value: Tensor("ReaderRead:1", shape=(), dtype=string)
    # decode_csv will convert a Tensor from type string (the text line) in a tuple of tensor columns with the specified defaults, which also sets the data type for each column
    decoded=tf.decode_csv(value,field_delim=' ',record_defaults=[[0] for i in range(FLAGS.max_sentence_len*2)])
    #print(len(decoded))
    #print(decoded)
    # len(decoded)=160 
    # decoded: [<tf.Tensor 'DecodeCSV:0' shape=() dtype=int32>, <tf.Tensor 'DecodeCSV:1' shape=() dtype=int32>, ... ]
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,batch_size=batch_size,capacity=batch_size*50,min_after_dequeue=batch_size)




def inputs(path):
    whole=read_csv(FLAGS.batch_size,path)
    #print(whole) # 160
    #print(whole.get_shape())
    # whole: [<tf.Tensor 'shuffle_batch:0' shape=(100,) dtype=int32>,
    features=tf.transpose(tf.pack(whole[0:FLAGS.max_sentence_len]))
    label=tf.transpose(tf.pack(whole[FLAGS.max_sentence_len:]))
    return features,label

def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)

def test_evaluate(sess,unary_score,test_sequence_length,transMatrix,inp,tX,tY):
    totalEqual=0
    batchSize=FLAGS.batch_size
    totalLen=tX.shape[0]
    # totalLen 18313 tX shape [18313,80]
    numBatch=int((tX.shape[0]-1)/batchSize)+1
    correct_labels=0
    correct_lines=0
    total_labels=0
    for i in range(numBatch):
        endOff=(i+1)*batchSize
        if endOff>totalLen:
            endOff=totalLen
        y=tY[i*batchSize:endOff]
        # y shape [100,80]
        feed_dict={inp: tX[i*batchSize:endOff]}
        unary_score_val,test_sequence_length_val=sess.run([unary_score,test_sequence_length],feed_dict=feed_dict)
        # unary_score_val shape [100,80,4] test_sequence_length_val shape [100]
        #print('unary_score_val.shape:',unary_score_val.shape)
        #print('test_sequence_length_val.shape:',test_sequence_length_val.shape)
        #print('y.shape:',y.shape)
        for tf_unary_scores_,y_,sequence_length_ in zip (unary_score_val,y,test_sequence_length_val):
            # tf_unary_scores_ shape [80,4] y_[80] sequence_length_ []
            #print('tf_unary_scores_.shape:',tf_unary_scores_.shape)
            #print('y_.shape:',y_.shape)
            #print('sequence_length_:',sequence_length_)
            tf_unary_scores_=tf_unary_scores_[:sequence_length_]
            # tf_unary_scores_ shape [
            y_=y_[:sequence_length_]
            viterbi_sequence,_=tf.contrib.crf.viterbi_decode(tf_unary_scores_,transMatrix)
            viterbi_sequence=np.asarray(viterbi_sequence)
            #print("viterbi_sequence")
            #print(viterbi_sequence)
            #print("viterbi_score:",_)
            # Evaluate word-level accuracy
            #print('y_:')
            #print(y_)
            viterbi_res=np.where(viterbi_sequence==15)
            #print("viterbi_res:")
            #print(viterbi_res)
            #print("y_res:")
            y_res=np.where(y_==15)
            #print(y_res)
            if len(viterbi_res[0])==len(y_res[0]):
                sum=np.sum(np.equal(viterbi_res,y_res))
                if sum==1 and len(y_res[0])==1:
                    #print("ok")
                    correct_lines+=1
            #print(np.equal(viterbi_sequence,y_))
            #sum=np.sum(np.equal(viterbi_sequence,y_))
            #print(sum)
            #if sum==sequence_length_:
                #print("ok")
                #correct_lines+=1
            #correct_labels+=np.sum(np.equal(viterbi_sequence,y_))
            #total_labels+=sequence_length_
    #accuracy=100.0*correct_labels/float(total_labels)
    accuracy=100.0*correct_lines/float(totalLen)
    print("Accuracy: %.2f%%" % accuracy)






def main(unused_argv):
    curdir=os.path.dirname(os.path.realpath(__file__))
    trainDataPath=tf.app.flags.FLAGS.train_data_path
    filenames=yfile.getFileDir(trainDataPath)
    print(filenames)
    graph=tf.Graph()
    with graph.as_default():
        model=Model(FLAGS.embedding_size,FLAGS.num_tags,FLAGS.word2vec_path,FLAGS.num_hidden)
        X,Y=inputs(filenames)
        #print(X)
        #print(Y)
        # X shape [100,80] Y shape[100,80]
        tX,tY=do_load_data(FLAGS.test_data_path)
        #print(tX)
        #print(tX.shape)
        # tX shape [18313,80]
        total_loss,loglike=model.loss(X,Y)
        train_op=train(total_loss)
        test_unary_score,test_sequence_length=model.test_unary_score()
        sv=tf.train.Supervisor(graph=graph,logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            training_steps=FLAGS.train_steps
            #training_steps=1
            for step in xrange(training_steps):
                if sv.should_stop():
                    break
                try:
                    _,trainsMatrix=sess.run([train_op,model.transition_params])
                    #print("crf_log_likelihood,return transition_params")
                    #print(trainsMatrix)
                    loss_val,loglike_val=sess.run([total_loss,loglike])
                    #print(loss_val)
                    #print(loglike_val)
                    #print(loglike_val.shape)
                    if step % 100==0:
                        print("[%d] loss:[%r]" % (step,sess.run(total_loss)))
                    if step % 1000==0:
                        test_evaluate(sess,test_unary_score,test_sequence_length,trainsMatrix,model.inp,tX,tY)
                except KeyboardInterrupt, e:
                    sv.saver.save(sess,FLAGS.log_dir+'/model',global_step=step+1)
                    raise e
            sv.saver.save(sess,FLAGS.log_dir+'/finnal-model')
            sess.close()


if __name__=='__main__':
    tf.app.run()








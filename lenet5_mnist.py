import os
import sys
import tensorflow as tf
import numpy as np
import pickle as pk
from tensorflow.examples.tutorials.mnist import input_data
home='/home/liuyang'

INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.ConfigProto()
config.gpu_options.allow_growth=True#增长式

# 配置神经网络的参数
BATCH_SIZE = 100
REGULARAZTION_RATE = 1e-4
TRAINING_STEPS = 30000
LABEL='relu'

def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [None,
                                    IMAGE_SIZE,  # 第一维表示一个batch中样例的个数
                                    IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸
                                    NUM_CHANNELS],  # 第四维表示图片的深度，对于RGB格式的图片，深度为5
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    dropout=tf.placeholder(tf.float32)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    def print_shape(t):
        print(t.op.name,' ',t.get_shape().as_list())
    def weight(shape):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    def bias(shape):
        return tf.Variable(tf.constant(0.1,shape=shape))
    def activate(x,b,label):  
        if label=='relu':
            return tf.nn.relu(x+b)
        
    w1=weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP])
    b1=bias([CONV1_DEEP])
    net1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME',name='conv1')
    print_shape(net1)
    net = activate(net1, b1, LABEL)
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool1')
    
    w2=weight([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
    b2=bias([CONV2_DEEP])
    net2 = tf.nn.conv2d(net, w2, strides=[1, 1, 1, 1], padding='SAME',name='con2')
    print_shape(net2)
    net = activate(net2, b2, LABEL)
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool2')

    #w3_=weight([2, 2, 64, 64])
    #b3_=bias([64])
    #net3 = tf.nn.conv2d(net, w3_, strides=[1, 1, 1, 1], padding='SAME',name='con3')
    #print_shape(net3)
    #net = activate(net3, b3_, LABEL)
    #net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool3')

    reshaped = tf.reshape(net, [-1, 3136])
    w3 = weight([3136, FC_SIZE])
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(w3))
    b3 = bias([FC_SIZE])
    net = activate(tf.matmul(reshaped, w3), b3, LABEL)
    net = tf.nn.dropout(net, dropout)
    
    w4 = weight([FC_SIZE, NUM_LABELS])
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(w4))
    b4 = bias([NUM_LABELS])
    
    logit = tf.matmul(net, w4,name='logit') + b4
    y = tf.nn.softmax(logit)
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y+1e-10),reduction_indices=[1]))
    train_step=tf.train.AdamOptimizer(REGULARAZTION_RATE).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          IMAGE_SIZE,
                                          IMAGE_SIZE,
                                          NUM_CHANNELS))
            _, loss_value, acc= sess.run([train_step, cross_entropy,accuracy], feed_dict={x: reshaped_xs, y_: ys, dropout: 0.5})
            if i%2000==0:
                print('training',i,'acc:',acc)
        
        test_accuracy=0.0
        for j in range(int(10000/BATCH_SIZE)):
            xs,ys=mnist.test.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          IMAGE_SIZE,
                                          IMAGE_SIZE,
                                          NUM_CHANNELS))
            acc=accuracy.eval(feed_dict={x:reshaped_xs,y_:ys, dropout: 1.0})
            test_accuracy+=acc   
        test_accuracy= test_accuracy/int(10000/BATCH_SIZE)
        print('test_accuracy:',test_accuracy)
        
        def get(n,label):
            reshaped_x = np.reshape(mnist.test.images[n],(1,28,28,1))
            reshaped_y = np.reshape(mnist.test.labels[n],(1,10))
            feed_dict = {x: reshaped_x,y_: reshaped_y, dropout:1.0}

            y_pre , y_label= sess.run([y,y_],feed_dict=feed_dict)
            y_prediction = np.reshape(y_pre,(10))
            y_prediction_label = np.reshape(y_label,(10))
            
            y_prediction = y_prediction.tolist()
            y_prediction_label = y_prediction_label.tolist()
            
            prediction = y_prediction.index(max(y_prediction))
            prediction_label = y_prediction_label.index(max(y_prediction_label))
            if prediction == prediction_label:
                if not os.path.isdir(home+'/save/%s/%d'%(label,n)):
                    os.makedirs(home+'/save/%s/%d'%(label,n))
                feature_map1 = net1.eval(feed_dict=feed_dict)
                f=open(home+'/save/%s/%d/feature_map1.pk'%(label,n),'wb')
                pk.dump(feature_map1,f)
                f.close()
                feature_map2 = net2.eval(feed_dict=feed_dict)
                f=open(home+'/save/%s/%d/feature_map2.pk'%(label,n),'wb')
                pk.dump(feature_map2,f)
                f.close()
                f=open(home+'/save/%s/%d/prediction_%d'%(label,n,prediction),'w')
                f.write('1')
                f.close()
                weight_ = w3.eval()
                f=open(home+'/save/%s/%d/weight.pk'%(label,n),'wb')
                pk.dump(weight_,f)
                f.close()
                
        for i in range(1000):
            get(i,'base')

        sys.exit()  
                    
def main(argv=None):
    mnist = input_data.read_data_sets("/home/liuyang/workspace/n_fold_superposition/src/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()

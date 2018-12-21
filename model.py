import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tflearn.layers.conv import global_avg_pool

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope



N_CLASSES=6
BATCH_SIZE=10
KEPP_PROB=0.8

def conv_2d(x, ksize,stride,filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    weights = tf.get_variable('weights',
                         shape=shape,
                         dtype='float32',
                         initializer=tf.contrib.layers.xavier_initializer(),
                         trainable=True)
    
    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)

    
    bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.5, dtype='float'))
    
    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, bias)
    
    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(x, bias)

def Relu(x):
    return tf.nn.relu(x)

def dropout(x):
    tf.nn.dropout(x, KEPP_PROB, noise_shape=None, seed=None,name=None) 

def Relu6(x):
    return tf.nn.relu6(x)
    
def Swish(x,beta=1):
  return x*tf.nn.sigmoid(x*beta)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def leaky_relu(x):
    alpha=0.2
    return tf.nn.leaky_relu(x,alpha)


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Avg_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = Global_Average_Pooling(input_x)
            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation
            return scale
        

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.6,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))



def inference(images, n_classes,training_flag):
    if training_flag=='training':
      is_training=tf.constant(True)
    if training_flag=='testing':
      is_training=tf.constant(False)
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    ksize=3
    stride=1
    filters_out=16
    ratio=4
    
    with tf.variable_scope("first_layer") :
        
        x=conv_2d(images, ksize,stride,filters_out)
        
        x=Squeeze_excitation_layer(x, filters_out, ratio, "first_layer")
        x = Batch_Normalization(x, training=is_training, scope="first_layer")
        
        x=Relu(x)
        #x=Max_pooling(x, pool_size=[3,3], stride=1, padding='SAME')
    
    for i in range(8):
        with tf.variable_scope("conv"+str(i)+'A') :
            short_cut=x
            #x=dropout(x)
            x=conv_2d(x, ksize,stride,filters_out)
            #x=Squeeze_excitation_layer(x, filters_out, ratio, "conv"+str(i))
            x = Batch_Normalization(x, training=is_training, scope="conv"+str(i))
            x=Relu(x)
            #x=Max_pooling(x, pool_size=[3,3], stride=1, padding='SAME')
        with tf.variable_scope("conv"+str(i)+'B') :
            
            #x=dropout(x)
            x=conv_2d(x, ksize,stride,filters_out)
            #x=Squeeze_excitation_layer(x, filters_out, ratio, "conv"+str(i))
            x = Batch_Normalization(x, training=is_training, scope="conv"+str(i))
            x=x+short_cut
            x=Relu(x)
            #x=Max_pooling(x, pool_size=[3,3], stride=1, padding='SAME')
        
    # full-connect1
    with tf.variable_scope("fc1"):
        reshape = layers.flatten(x)
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = Relu(tf.matmul(reshape, weights) + biases)

    # full_connect2
    with tf.variable_scope("fc2"):
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = Relu(tf.matmul(fc1, weights) + biases)

    # softmax
    with tf.variable_scope("softmax_linear") :
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
        print("the size of softmax_linear ")
        print(softmax_linear.shape)
        # softmax_linear = tf.nn.softmax(softmax_linear)

    return softmax_linear



def value2type(value):
    if value==0:
        return 'OCEAN'
    if value==1:
        return 'MOUNTAIN'
    if value==2:
        return 'FARMLAND'
    if value==3:
        return 'LAKE'
    if value==4:
        return 'CITY'
    if value==5:
        return 'DESERT'
    
def value2onehot(value):
    list1=[]
    if value==0:
        list1=[1,0,0,0,0,0]
    if value==1:
        list1=[0,1,0,0,0,0]
    if value==2:
        list1=[0,0,1,0,0,0]
    if value==3:
        list1=[0,0,0,1,0,0]
    if value==4:
        list1=[0,0,0,0,1,0]
    if value==5:
        list1=[0,0,0,0,0,1]
    return list1


def losses(logits, labels):
    with tf.variable_scope('loss'):
        print("loss")
        print("check the size of logits")
        print(logits.shape)
        print("check the size of labels")
        print(labels.shape)
    
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        loss = tf.reduce_mean(cross_entropy)
    return loss


def evaluation(logits, labels):
    with tf.variable_scope("accuracy"):
        print("accuracy")
        print("check the size of logits")
        print(logits.shape)
        print("check the size of labels")
        print(labels.shape)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    tf.reset_default_graph()
    IMG_SIZE = 200
    BATCH_SIZE = 1
    CAPACITY = 200
    N_CLASSES = 6
    pwd=os.getcwd()
    logs_dir = pwd+'\\logs_1\\'     # 检查点保存路径
    

    # 测试图片读取
    test_image='MWI_jV4PcBehzPRH2lwi.jpg'
    test_image = tf.read_file(test_image)
    test_image = tf.image.decode_jpeg(test_image, channels=3)
    test_image = tf.image.resize_image_with_crop_or_pad(test_image, IMG_SIZE, IMG_SIZE)
    test_image = tf.cast(test_image, tf.float32)
    test_image=tf.reshape(test_image,[BATCH_SIZE,IMG_SIZE,IMG_SIZE,3])
    #test_label=tf.constant(5)
    #test_label=tf.reshape(test_label,[BATCH_SIZE,1])
    test_label=tf.constant(value2onehot(5))
    test_label=tf.reshape(test_label,[BATCH_SIZE,N_CLASSES])
    
    
    logits=inference(test_image, N_CLASSES,'testing')
    softmax=tf.nn.softmax(logits)
    predict=tf.argmax(softmax, 1)
    predict2=predict[0]
    
    

    loss=losses(logits, test_label)
    acc=evaluation(logits, test_label)
    
    sess = tf.Session()
    
    sess.run(tf.initialize_all_variables())
    
    acc_, loss_ = sess.run([acc, loss])
    logits_,test_label_,softmax_,predict_=sess.run([logits, test_label,softmax,predict2])
    
    predict_=value2onehot(predict_)
    
    print("acc_")
    print(acc_)
        
    print("loss_")
    print(loss_)
    
    print("softmax_")
    print(softmax_)
    
    print("test_label_")
    print(test_label_)
    
    print("predict_")
    print(predict_)
    
    


            
        


    
    
    


    

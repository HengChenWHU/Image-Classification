import time
from load_data import *
from model import *
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

pwd=os.getcwd()
train_dir = pwd+'\\data\\train_2000\\train_images\\train_set\\'
allimages=pwd+'\\data\\train_2000\\train_images\\allimages\\'
devlopment_set='./data/train_2000/train_images/development/'
logs_train_dir =pwd+ '\\data\\train_2000\\train_images\\train_label_2000_new.txt'
logs_dir = pwd+'\\logs_1\\'     # 检查点保存路径

# 训练模型
def training():
    N_CLASSES = 6
    IMG_SIZE = 50
    BATCH_SIZE = 50
    CAPACITY = 200
    MAX_STEP = 10000
    LEARNING_RATE = 1e-4
    MOVING_AVERAGE_DECAY = 0.99
    UPDATE_OPS_COLLECTION = 'update_ops'
    weight_decay=0.95
   
    p=tf.constant(value=[0.064246,0.064103,0.133897,0.520980,0.520980,0.046216])
    training_flag = tf.placeholder(tf.bool)#
    
    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0),trainable=False)#创建global_step参数

    #产生一个会话
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 占用GPU90%的显存 
    config.allow_soft_placement=True
    config.log_device_placement=True
    config.gpu_options.allocator_type='BFC'
    sess = tf.Session()

    train_image_list,train_label_list,dic1 = get_all_files(train_dir, logs_train_dir)#this sentence form a list of all image dir and label dir
    image_train_batch,label_train_batch = get_batch(train_image_list,train_label_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    
    image_train_batch=tf.cast(image_train_batch,dtype=tf.float32)
    print("label_train_batch")
    print(label_train_batch.shape)
    
    
    image_train_batch=progress_picture(image_train_batch,IMG_SIZE)#preprogram argument picture
    
    train_logits = inference(image_train_batch, N_CLASSES,'training')
    
    train_logits=tf.multiply(train_logits,p)
    
    train_loss = losses(train_logits, label_train_batch)
    tf.summary.scalar('loss',train_loss)

    #regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
    #reg_term = tf.contrib.layers.apply_regularization(regularizer)
    
    #train_loss=train_loss+reg_term
    
    ########################################################
    # loss_avg
    #ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step) #定义一个滑动平均的类
    #ave_loss=ema.apply([train_loss]) #对loss_进行滑动平均
    
    ###########################################################
    
    #l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    
    ######################################退化学习率######################################
    decay_learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,decay_steps=1000,decay_rate=0.5)
    

    #train_op = tf.train.AdamOptimizer(decay_learning_rate).minimize(train_loss+l2_loss * weight_decay)
    ######################################采用adam优化函数######################################
    train_op = tf.train.AdamOptimizer(decay_learning_rate).minimize(train_loss)
     ######################################计算一个Batch的训练精度######################################
    train_acc = evaluation(train_logits, label_train_batch)
    tf.summary.scalar('accurate',train_acc)

    ######################################计算参数的数目######################################
    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目:%d' % sess.run(paras_count), end='\n\n')

    saver = tf.train.Saver()

    add_global = global_step.assign_add(1)
    
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
        sess.run(tf.global_variables_initializer())
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("val_record/", sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            
            rate,g=sess.run([decay_learning_rate,add_global])

            _, loss, acc = sess.run([train_op, train_loss, train_acc])
            #reg_term_=sess.run(reg_term)
            
            if step % 2 == 0:  # 实时记录训练过程并显示
                runtime = time.time() - s_t
                print('Step: %6d, loss: %.8f, accuracy: %.2f, learning_rate:%.8f ,time:%.4f'% (g, loss, acc * 100,rate,runtime))
                rs=sess.run(merged)
                writer.add_summary(rs, step)
                s_t = time.time()

            if step % 200 == 0 or step == MAX_STEP - 1:  # 保存检查点
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("have already saved the model")
                #eval()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()

def eval():
    N_CLASSES = 6
    IMG_SIZE = 50
    BATCH_SIZE = 1
    CAPACITY = 200
    MAX_STEP = 1
    
    p=tf.constant(value=[0.064246,0.064103,0.133897,0.520980,0.520980,0.046216])

    logs_dir = './logs_1/'    # 检查点目录

    sess = tf.Session()

    train_image_list,train_label_list,dic1 = get_all_files(devlopment_set,logs_train_dir)
    n=len(train_image_list)
    print("all images are %.3d"%(n))
    
    image = tf.placeholder(tf.float32, [BATCH_SIZE,IMG_SIZE,IMG_SIZE, 3], name='image')
    
    
    logits = inference(image, N_CLASSES,'testing')
    logits=tf.multiply(logits,p)
    
    prediction = tf.nn.softmax(logits)  # 用softmax转化为百分比数值
    
    max_index = tf.argmax(prediction[0],axis=0)
    #print("aaa")
    #print(max_index.shape)
    

    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
        
    correct_rate=0.00
    for i in range(n):
        #print(train_image_list[i])
        img=cv2.imread(train_image_list[i])
        #print(img)
        #print(train_image_list[i])
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img=[img]
        
    
        img=np.asarray(img)
        
        max_index_,prediction_= sess.run([max_index,prediction],feed_dict={image:img})
        
        #print("prediction_")
        #print(prediction_)
        
        #print("max_index_")
        #print(max_index_)
        #
        
        label=np.argmax(train_label_list[i])
        
        #print("label")
        #print(label)
        ifcor=""
        if max_index_==label:
            ifcor="Correct"
        else:
            ifcor="Wrong"
            
        print("test%d,predict:%d,and label:%d"%(i,max_index_,label))
        
        if max_index_==label:
            correct_rate+=1
    correct_rate=correct_rate/n
    print("最终的准确率为：%f"%(correct_rate))
    sess.close()

    
# 测试检查点
def test():
    N_CLASSES = 6
    IMG_SIZE = 50
    BATCH_SIZE = 1000
    CAPACITY = 2000
    MAX_STEP = 1

    logs_dir = 'logs_1'     # 检查点目录
    test_dir=pwd+'\\data\\test_set_1000\\'
    output='logs_1\\ouput.txt'
    f=open(output,'w')
    sess = tf.Session()

    test_list,test_img_name = get_all_files_test(test_dir)
    test_batch= get_batch_test(test_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    train_logits = inference(test_batch, N_CLASSES,'testing')
    prediction = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值
    
    max_index = tf.argmax(prediction,axis=1)
    max_index=tf.reshape(max_index,[BATCH_SIZE])

    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    correct_rate=0.00
    index=0
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            image, max_index_ = sess.run([test_batch, max_index])
            print("predict")
            print(max_index_.shape)
            for index in range(BATCH_SIZE):
                print(test_img_name[index]+','+value2type(max_index_[index]),file=f)
        print("---------------finished testing------------")
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
    f.close()


if __name__ == '__main__':
    tf.reset_default_graph()
    
    training()
    eval()
    #test()
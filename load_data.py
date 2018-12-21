import tensorflow as tf
import numpy as np
import os
import operator


    
def get_all_files(train_dir, logs_train_dir):
    """
    获取图片路径及其标签
    :param train_dir: a sting, 图片所在目录
    :param logs_train_dir: a sting, 标签所在路径
    :return:image_list,label_list2,dic1
    """
    image=[]#用于存储图片文件地址列表
    image_label=[]
    image_name=[]
    key=[]
    value=[]
    
    for line in open(logs_train_dir):   
        temp=line.split(',')
        key.append(temp[0])
        value.append(temp[1])
    dic1 = dict(map(lambda x,y:[x,y],key,value))
    

    for file in os.listdir(train_dir):
        image.append(train_dir + '/' + file)
        image_name.append(file)
        #print(file)
        #print(dic1[file])
        image_label.append(dic1[file])
    dic2 = dict(map(lambda x1,y1:[x1,y1],image_name,image_label))



    image_list = np.asarray(image)
    label_list = np.asarray(image_label)
    #print(label_list)
    #we must be sure that the label_list is interger typr
    label_list = [int(i) for i in image_label]
    """
    count0=0
    count1=0
    count2=0
    count3=0
    count4=0
    count5=0
    
    for i in label_list:
        if i==0:
            count0=count0+1
        if i==1:
            count1=count1+1
        if i==2:
            count2=count2+1
        if i==3:
            count3=count3+1
        if i==4:
            count4=count4+1
        if i==5:
            count5=count5+1
            
    allimage=count0+count1+count2+count3+count4+count5
    p0=allimage/count0
    p1=allimage/count1
    p2=allimage/count2
    p3=allimage/count3
    p4=allimage/count4
    p5=allimage/count5
    
    p_all=p0+p1+p2+p3+p4+p5
    p0=float(p0/p_all)
    p1=float(p1/p_all)
    p2=float(p2/p_all)
    p3=float(p4/p_all)
    p4=float(p4/p_all)
    p5=float(p5/p_all)
    
    
    print("OCEAN  %d  P=: %f.4"%(count0,p0))
    print("MOUNTAIN  %d  P=: %f.4"%(count1,p1))
    print("FARMLAND  %d  P=: %f.4"%(count2,p2))
    print("LAKE  %d  P=: %f.4"%(count3,p3))
    print("CITY  %d  P=: %f.4"%(count4,p4))
    print("DESERT  %d  P=: %f.4AA"%(count5,p5))
    """
        
            
    
    label_list2 = [value2onehot(i) for i in label_list]
     
    #return image_list,label_list,dic1
    #train_image_list=image_list[:1800]
    #dev_image_list=label_list[1800:]
    
    #train_label_list=label_list[:1800]
    #dev_label_list=label_list[1800:]
    #print((train_image_list))
    #print((dev_image_list))
    #print((train_label_list))
    #print((dev_label_list))
    return image_list,label_list2,dic1

def get_all_files_test(train_dir):
    """
    获取图片路径及其标签
    :param file_path: a sting, 图片所在目录
    :param is_random: True or False, 是否乱序
    :return:
    """
    image=[]#用于存储图片文件地址列表
    image_name=[]
    for file in os.listdir(train_dir):
        image.append(train_dir + '/'+file)
        image_name.append(file)
    image_list = np.asarray(image)
    image_name = np.asarray(image_name)
    return image_list,image_name
    
def progress_picture(image_train,image_size):
    #数据加强过程
    #image_train=tf.random_crop(image_train,[image_size,image_size,3])
    
    #tf.image.random_flip_left_right(image_train)#随机左右翻转
    
    #tf.image.random_flip_up_down(image_train)#随机上下翻转
    
    #image_train = tf.image.resize_image_with_crop_or_pad(image_train, image_size, image_size)
    #随机设置图片的亮度
    image_train = tf.image.random_brightness(image_train,max_delta=1)
    #随机设置图片的对比度
    image_train = tf.image.random_contrast(image_train,lower=0.2,upper=1.8)
    
    #随机设置图片的色度
    image_train = tf.image.random_hue(image_train,max_delta=0.3)
    
    #随机设置图片的饱和度
    image_train = tf.image.random_saturation(image_train,lower=0.2,upper=1.8)
    # 图像标准化，
    #image_train = tf.image.per_image_standardization(image_train)
    image_train = tf.cast(image_train, tf.float32)  # 转换数据类型并归一化
    
    return image_train
    
    


def get_batch(train_list,label_list, image_size, batch_size, capacity, is_random=True):
    """
    获取训练批次
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, 训练图像大小
    :param batch_size: a int, 每个批次包含的样本数量
    :param capacity: a int, 队列容量
    :param is_random: True or False, 是否乱序
    :return:
    """

    intput_queue = tf.train.slice_input_producer([train_list,label_list], shuffle=True)
    
    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # 这里是jpg格式
    
    image_train = tf.image.resize_image_with_crop_or_pad(image_train, image_size, image_size)
    
    #image_train=progress_picture(image_train,image_size)

    # 图片标签
    label_train = intput_queue[1]

    # 获取批次
    if is_random:
        image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train],
                                                                      batch_size=batch_size,
                                                                      capacity=capacity,
                                                                      min_after_dequeue=100,
                                                                      num_threads=2)
    else:
        image_train_batch, label_train_batch = tf.train.batch([image_train, label_train],
                                                              batch_size=1,
                                                              capacity=capacity,
                                                              num_threads=1)
    return image_train_batch, label_train_batch


def get_batch_eval(train_list,label_list, image_size, batch_size, capacity, is_random=True):
    """
    获取训练批次
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, 训练图像大小
    :param batch_size: a int, 每个批次包含的样本数量
    :param capacity: a int, 队列容量
    :param is_random: True or False, 是否乱序
    :return:
    """

    intput_queue = tf.train.slice_input_producer([train_list,label_list], shuffle=True)
    
    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # 这里是jpg格式
    
    image_train = tf.image.resize_images(image_train, [image_size, image_size])
    
    image_train = tf.cast(image_train, tf.float32)  # 转换数据类型并归一化

    # 图片标签
    label_train = intput_queue[1]

    # 获取批次
    if is_random:
        image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train],
                                                                      batch_size=batch_size,
                                                                      capacity=capacity,
                                                                      min_after_dequeue=100,
                                                                      num_threads=2)
    else:
        image_train_batch, label_train_batch = tf.train.batch([image_train, label_train],
                                                              batch_size=1,
                                                              capacity=capacity,
                                                              num_threads=1)
    return image_train_batch, label_train_batch


def get_batch_test(train_list, image_size, batch_size, capacity, is_random=True):
    """
    获取训练批次
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, 训练图像大小
    :param batch_size: a int, 每个批次包含的样本数量
    :param capacity: a int, 队列容量
    :param is_random: True or False, 是否乱序
    :return:
    """

    intput_queue = tf.train.slice_input_producer([train_list], shuffle=False)
    
    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # 这里是jpg格式
    
    image_train = tf.image.resize_images(image_train, [image_size, image_size])
    
    image_train = tf.cast(image_train, tf.float32)  # 转换数据类型并归一化


    # 获取批次
    if is_random:
        image_train_batch = tf.train.shuffle_batch([image_train],
                                                  batch_size=batch_size,
                                                  capacity=capacity,
                                                  min_after_dequeue=100,
                                                  num_threads=2)
    else:
        image_train_batch = tf.train.batch([image_train],
                                           batch_size=1,
                                           capacity=capacity,
                                           num_threads=1)
    return image_train_batch

def onehot2type(label):
    print(label)
    print(label.shape)
    OCEAN=[1,0,0,0,0,0]
    MOUNTAIN=[0,1,0,0,0,0]
    FARMLAND=[0,0,1,0,0,0]
    LAKE=[0,0,0,1,0,0]
    CITY=[0,0,0,0,1,0]
    DESERT=[0,0,0,0,0,1]
    if operator.eq(label.all(),OCEAN):
        return 'OCEAN'
    if operator.eq(label,MOUNTAIN):
        return 'MOUNTAIN'
    if operator.eq(label,FARMLAND):
        return 'FARMLAND'
    if operator.eq(label,LAKE):
        return 'LAKE'
    if operator.eq(label,CITY):
        return 'CITY'
    if operator.eq(label,DESERT):
        return 'DESERT'
    
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    pwd=os.getcwd()
    IMG_SIZE = 200
    BATCH_SIZE = 1
    CAPACITY = 200
    logs_train_dir = 'E:/Study/big_competition/tiangong/train_2000/train_images/train_label_2000_new.txt'
    train_dir = pwd+'\\data\\train_2000\\train_images\\train_set\\'
    

    # 测试图片读取
    train_image_list,train_label_list,dic1 = get_all_files(train_dir, logs_train_dir)
    
    #label_list = [value2onehot(i) for i in train_label_list]
    
    image_train_batch, label_train_batch = get_batch(train_image_list,train_label_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)

    sess = tf.Session()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(4):
            if coord.should_stop():
                break
            image_batch, label_batch = sess.run([image_train_batch, label_train_batch])
            #print("label")
            #print(label_batch[0])
            for batch in range(BATCH_SIZE):
                plt.imshow(image_batch[batch])#, plt.title(label)
                print(label_batch[0])
                #print(value2type(label_batch[batch]))
                #print(value2onehot(label_batch[batch]))
                plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
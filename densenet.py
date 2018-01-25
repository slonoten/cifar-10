
import sys
import csv
import numpy as np
from keras.datasets import cifar10
from random import randrange
import time
import os
import math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d

def get_log_dir(base_dir):
    def num(s):
        try:
            return int(s)
        except ValueError:
            return None
    
    numeric_dirs = [num(name) for name in os.listdir(base_dir) if os.path.isdir(base_dir + '/' + name)]
    return base_dir + '/' + str(max([n for n in numeric_dirs if not n is None] + [0]) + 1)


def augment_chrominance(image,
                        hue_max_delta=0.05,
                        contrast_lower=0.3, contrast_upper=1.0,
                        brightness_max_delta=0.2,
                        saturation_lower=0.0, saturation_upper=2.0):
        #image = tf.image.random_hue(image, max_delta=hue_max_delta)
        image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        #image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
        return image   

def augment(images, labels,
            resize=None, # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0, # Maximum rotation angle in degrees
            crop_probability=0, # How often we do crops
            crop_min_percent=0.6, # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
      
    with tf.variable_scope("augmentation"):
        if resize is not None:
            images = tf.image.resize_bilinear(images, resize)
        
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                    tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                    tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                    tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                    tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
            tf.contrib.image.angles_to_projective_transforms(
                angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent, crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            crop_transform = tf.stack([
                  crop_pct,
                  tf.zeros([batch_size]), top,
                  tf.zeros([batch_size]), crop_pct, left,
                  tf.zeros([batch_size]),
                  tf.zeros([batch_size])
                ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                    tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR') # or 'NEAREST'

        def cshift(values): # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            lm = tf.expand_dims(lam, -1)
            images = ll * images + (1 - ll) * cshift(images)
            labels = tf.multiply(lm, labels, 'a') + tf.multiply((1 - lm), cshift(labels), 'b')

    return images, labels

def build_dense_slim(hyperparams, total_layers = 25):
    def conv2d(input_node, channels_n, level, is_training, stride = 1, activation = True):
        input_shape = input_node.get_shape().as_list()
        assert len(input_shape) == 4, 'Tensor with rank 4 is expected.'
        in_channels_n = input_shape[3]
        with tf.variable_scope(f'conv_{level}'):
            Wconv1 = tf.get_variable(f"W_{level}", shape=[3, 3, in_channels_n, channels_n], initializer= xavier_initializer_conv2d())
            bconv1 = tf.get_variable(f"b_{level}", shape=[channels_n], initializer=tf.zeros_initializer)  
            conv = tf.nn.conv2d(input_node, Wconv1, strides=[1,stride,stride,1], padding='SAME') + bconv1
            batch_norm = tf.layers.batch_normalization(conv, training = is_training)
            output = tf.nn.relu(batch_norm) if activation else batch_norm
        return output    
            
    def denseBlock(input_layer, i, j, is_training):
        with tf.variable_scope(f"dense_unit_{i}"):
            nodes = []
            a = conv2d(input_layer, 64, f"{i}_0", is_training)
            nodes.append(a)
            for z in range(j):
                b = conv2d(tf.concat(nodes, 3), 64, f"{i}_{z+1}", is_training)
                nodes.append(b)
            return b

    units_between_stride = int(total_layers / 5)
    input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
    label_layer = tf.placeholder(shape=[None],dtype=tf.int64) 
    label_oh = tf.one_hot(label_layer, 10, dtype=tf.float32)
    is_train = tf.placeholder(tf.bool, shape = (), name = 'is_train')
    
    def aug(): 
        images = tf.map_fn(lambda img: augment_chrominance(img), input_layer)
        return augment(images, label_oh, horizontal_flip=True, rotate=15, crop_probability=0.8, mixup=hyperparams['mixup_alpha'])
    
    input_layer_aug, label_oh_aug = tf.cond(is_train, aug, lambda : (input_layer, label_oh))

    layer1 = conv2d(input_layer_aug, 64, 'conv_0', is_train)
    for i in range(5):
        layer1 = denseBlock(layer1, i, units_between_stride, is_train)
        layer1 = conv2d(layer1, 64, f"{i}_{units_between_stride}", is_train, stride = 2)

    top = conv2d(layer1, 10, 'top', is_train, activation = False)
    
    flat_top = tf.reshape(top, [-1, 10])

    output = tf.nn.softmax(flat_top)

    loss = tf.reduce_mean(-tf.reduce_sum(label_oh_aug * tf.log(output) + 1e-10, reduction_indices=[1]))
    
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(label_oh_aug, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (input_layer, label_layer, is_train), (output, loss, accuracy)

def build_test_summary():
    loss_value = tf.placeholder(tf.float64, shape = (), name = 'summary_loss')
    accuracy_value = tf.placeholder(tf.float64, shape = (), name = 'summary_accuracy')
    with tf.name_scope('summaries'):
        ml = tf.summary.scalar('mean_loss', loss_value)
        acc = tf.summary.scalar('accuracy', accuracy_value)
    summary = tf.summary.merge([ml, acc])
    return (loss_value, accuracy_value), summary 


def training_loop(session, saver, model_inputs, model_outputs, train_step, epochs=10, batch_size=64):
    # Инициализируем tensorboard
    log_dir_base = get_log_dir('/home/max/ipython/tmp/tensorboard')
    writer_train = tf.summary.FileWriter(log_dir_base + '/train', graph=tf.get_default_graph())
    writer_test = tf.summary.FileWriter(log_dir_base + '/test')
    #создаём индекс по всем объектам
    index = np.arange(len(x_train))
    
    #перемешиваем его
    np.random.shuffle(index)
    
    #разбиваем на батчи
    num_batches = int(len(index) / batch_size)
    batch_indexes = np.array_split(index, num_batches)
    
    #аналогично для теста
    index_test = np.arange(len(x_test))
    np.random.shuffle(index_test)
    num_batches_test = int(len(index_test) / batch_size)
    batch_indexes_test = np.array_split(index_test, num_batches_test)
    
    #аналогично для validation
    index_val = np.arange(len(x_val))
    np.random.shuffle(index_val)
    num_batches_val = int(len(index_val) / batch_size)
    batch_indexes_val = np.array_split(index_val, num_batches_val)
    
    model_file = None
        
    x, y, is_train, loss_val, acc_val = model_inputs
    y_out, mean_loss, accuracy, summary = model_outputs
    
    def train(x_values, y_values, batch_indexes, epoch):
        train_loses = []
        for i, batch_index in enumerate(batch_indexes):

            #Создаём словарь, осуществляющий сопоставление входов графа (plaseholders) и значений
            feed_dict = {is_train: True,
                         x: x_values[batch_index],
                         y: y_values[batch_index]}

            #Здесь происходит непоследственный вызов модели
            #Обратите внимание, что мы передаём train_step
            scores, loss, acc, _ = session.run([y_out, mean_loss, accuracy, train_step],feed_dict=feed_dict)
            s, = session.run([summary], feed_dict = { loss_val : loss, acc_val : acc })
            writer_train.add_summary(s, epoch * num_batches + i)
            train_loses.append(loss)
            print(f'iteration {i}, train loss: {loss:.3}, accuracy: {acc:.3}', end='\r')
        return train_loses
        
    def evaluate(x_values, y_values, batch_indexes):
        test_loses = []
        test_accuracy = []

        for batch_index in batch_indexes:

            #Создаём словарь, осуществляющий сопоставление входов графа (plaseholders) и значений
            feed_dict = {is_train: False,
                         x: x_values[batch_index],
                         y: y_values[batch_index]}
            #Здесь происходит непоследственный вызов модели
            loss, acc = session.run([mean_loss, accuracy],feed_dict=feed_dict)

            test_loses.append(loss)
            test_accuracy.append(acc)

        return test_loses, test_accuracy
    
    # цикл по эпохам
    for e in range(epochs):
        print(f'Epoch {e}:')
        t = time.process_time()
        train_loses = train(x_train, y_train, batch_indexes, e)
        val_loses, val_accuracy = evaluate(x_val, y_val, batch_indexes_val)
        mean_val_loses = np.mean(val_loses)
        mean_val_accuracy = np.mean(val_accuracy)        
        print(f'train loss: {np.mean(train_loses):.3}, val loss: {mean_val_loses:.3}, accuracy: {mean_val_accuracy:.3}, time for epoch: {time.process_time() - t:.3}')
        s, = session.run([summary], feed_dict = { loss_val : mean_val_loses, acc_val : mean_val_accuracy })
        writer_test.add_summary(s, (e + 1) * num_batches - 1) 
        if e % 10 == 0:
            model_file = saver.save(session, "./my-model")
            test_loses, test_accuracy = evaluate(x_test, y_test, batch_indexes_test)
            summary_str = f'Epoch: {e}, test loss: {np.mean(test_loses):.3}, accuracy: {np.mean(test_accuracy):.3}'
            print(summary_str)
            
    model_file = saver.save(session, "./my-model")
    print('================================================')
    print('Test set results:')
    test_loses, test_accuracy = evaluate(x_test, y_test, batch_indexes_test)
    writer_train.close()
    writer_test.close()
    summary_str = f'test loss: {np.mean(test_loses):.3}, accuracy: {np.mean(test_accuracy):.3}'
    print(summary_str)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Нет параметров - тренировка с нуля.
        restore = False
        test = False
    else:
        restore = sys.argv[1] in ('test', 'restore')
        test = sys.argv[1] == 'test'
        
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255 - 0.5
    x_test = x_test / 255 - 0.5
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1000)

    #Перед вызовом функции очистим память от графов других моделей (актуально если вы вызываете эту ячейку повторно)
    tf.reset_default_graph()
    hyperparams = { 'mixup_alpha' : 0.5 }
    (x, y, is_train), (y_out, mean_loss, accuracy) = build_dense_slim(hyperparams, total_layers = 40)
    (loss_val, acc_val), summary = build_test_summary()

    #Теперь зададим алгоритм оптимизации
    optimizer = tf.train.AdamOptimizer(0.00001) 
    #train_step -- специальный служебный узел в графе, отвечающий за обратный проход
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss) 

    saver = tf.train.Saver()
    #send_telegram('468013605:AAFEsfZA-TsZ6sNzQwdY-70smTDhx6PBH3U', f"Start training with params {hyperparams}")
    # создаём сессию. Сессия -- это среда, в которой выполняются вычисления
    with tf.Session() as sess:
        #мы можем явно указать устройство
        with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0" 
            #инициализируем веса, в этот момент происходит выделение памяти
            if restore:
                print('Restoring saved model.')
                saver.restore(sess, "./my-model")
            else:
                sess.run(tf.global_variables_initializer())
            if test:
                print('Testing.')
                indexes = np.arange(len(x_test))
                index_batches = np.split(indexes, 100)
                with open('cifar-10.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)            
                    writer.writerow(['id', 'label'])        
                    for index_batch in index_batches:
                        feed_dict = {is_train: False,
                                    x: x_test[index_batch]}
                        y_tst = sess.run([y_out],feed_dict=feed_dict)
                        for (i, out) in zip(index_batch, y_tst[0]):
                            writer.writerow([i, np.argmax(out)])
            else:
                #запускаем тренировку
                print('Training.')
                training_loop(sess, saver, model_inputs=(x, y, is_train, loss_val, acc_val), 
                            model_outputs=(y_out, mean_loss, accuracy, summary), 
                            train_step=train_step, epochs=300, batch_size = 64)

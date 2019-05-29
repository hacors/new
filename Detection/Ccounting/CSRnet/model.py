import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from matplotlib import pyplot as plt
import process

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.enable_eager_execution()
KL = keras.layers
VGG16 = keras.applications.vgg16.VGG16
feature = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'img': tf.FixedLenFeature([], tf.string),
    'dens': tf.FixedLenFeature([], tf.string),
}


def parse_image_function(example_proto):  # 解码
    return tf.parse_single_example(example_proto, feature)


def process_function(parsed_data):
    height = parsed_data['height']
    width = parsed_data['width']
    img_string = parsed_data['img']
    dens_string = parsed_data['dens']
    img_true = tf.reshape(tf.decode_raw(img_string, tf.uint8), [height, width, 3])
    dens_true = tf.reshape(tf.decode_raw(dens_string, tf.float32), [height, width, 1])  # 注意图片必须是三维的
    img_processed = tf.divide(tf.cast(img_true, tf.float32), 255.0)
    img_expand = tf.expand_dims(img_processed, -1)
    img_part_0 = tf.divide(tf.subtract(img_expand[:, :, 0, :], 0.485), 0.229)
    img_part_1 = tf.divide(tf.subtract(img_expand[:, :, 1, :], 0.456), 0.224)
    img_part_2 = tf.divide(tf.subtract(img_expand[:, :, 2, :], 0.406), 0.225)
    img_processed = tf.concat([img_part_0, img_part_1, img_part_2], 2)
    dens_processed = tf.image.resize_images(dens_true, [height/8, width/8], method=1)*64  # 平衡数值大小，同时使得图像可以显示
    return img_processed, dens_processed


def euclidean_distance_loss(y_true, y_pred):
    loss_1 = keras.losses.mean_squared_error(y_true, y_pred)  # 注意对图片来说，loss针对的是图中的每一个像素点
    loss_2 = tf.sqrt(tf.reduce_sum(loss_1, axis=[1, 2]))
    loss_3 = tf.reduce_mean(loss_2, axis=0)
    return loss_3


def crowd_net():
    init = keras.initializers.RandomNormal(stddev=0.01)
    vgg = VGG16(weights='imagenet', include_top=False)  # W
    input_data = keras.Input(shape=(None, None, 3))
    crop_vgg = keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv3').output)  # 注意这是模型截取的写法
    digits = crop_vgg(input_data)
    digits = KL.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=init, padding='same')(digits)
    prediction = digits
    crowd_net = keras.Model(inputs=input_data, outputs=prediction)
    return crowd_net


def save_model(model: keras.Model, w_h5_path, json_path):
    model.save_weights(w_h5_path)
    model_json_data = model.to_json()
    with open(json_path, 'w') as json_file:
        json_file.write(model_json_data)


def summary_numpy(scatted_np: np.array):
    scatted_np = scatted_np.squeeze()
    merge_left = np.concatenate((scatted_np[0], scatted_np[1]), axis=0)
    merge_right = np.concatenate((scatted_np[2], scatted_np[3]), axis=0)
    result = np.concatenate((merge_left, merge_right), axis=1)
    return result


def show(img_array):
    img_array = img_array.squeeze()
    temp_array = img_array*255.0
    temp_array = temp_array.astype(np.uint8)
    plt.imshow(temp_array)
    plt.show()


if __name__ == "__main__":
    shtech_image_path, shtech_set_path = process.get_shtech_path()
    tfrecord_path = os.path.join(shtech_set_path[0][0], 'all_data.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(parse_image_function)
    processed_dataset = parsed_dataset.map(process_function)
    batched_dataset = processed_dataset.batch(1)  # 每个batch都是同一张图片切出来的
    mynet = crowd_net()
    # print(mynet.summary())
    for epoch in range(400):
        epoch_loss = list()
        for index, dataset in enumerate(batched_dataset):

            for repeat in range(20):
                with tf.GradientTape() as train_tape:
                    opti = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
                    predict = mynet(dataset[0], training=True)  # 注意所有的keras模型必须添上一句话，training=True
                    loss = euclidean_distance_loss(dataset[1], predict)
                    gradiens = train_tape.gradient(loss, mynet.variables)
                    opti.apply_gradients(zip(gradiens, mynet.variables))
                    if repeat % 5 == 0:
                        temp_img = dataset[0][0].numpy()
                        temp_dens_true = dataset[1][0].numpy()
                        temp_dens_pred = predict[0].numpy()
                        '''
                        show(temp_img)
                        show(temp_dens_pred)
                        show(temp_dens_true)
                        '''
                        print('loss:', loss.numpy(), 'true_max:', temp_dens_true.max(), 'true_mean', np.mean(temp_dens_true), 'max:',
                              temp_dens_pred.max(), 'min:', temp_dens_pred.min(), 'diff:', temp_dens_pred.max()-temp_dens_pred.min())

        if epoch % 5 == 0:
            mynet.save_weights('Datasets/shtech/weight_%s_new.h5' % epoch)
    save_model(mynet, 'Datasets/shtech/weight_last.h5', 'Datasets/shtech/model.json')

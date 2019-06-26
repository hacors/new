import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

import dataset
tf.enable_eager_execution()
BATCHSIZE = 1
KL = keras.layers
init = keras.initializers.RandomNormal(stddev=0.01)
model_path = 'Datasets/mall/model/'
feature = {
    'pic': tf.FixedLenFeature([], tf.string),
    'dens': tf.FixedLenFeature([], tf.string),
}


def parse_image_function(example_proto):  # 解码
    return tf.parse_single_example(example_proto, feature)


def process_function(parsed_data):
    pic_string = parsed_data['pic']
    dens_string = parsed_data['dens']
    pic_true = tf.reshape(tf.decode_raw(pic_string, tf.uint8), [dataset.video_length, dataset.crop_size, dataset.crop_size, 3])
    pic_true = tf.cast(pic_true, tf.float32)
    dens_true = tf.reshape(tf.decode_raw(dens_string, tf.float32), [dataset.video_length, dataset.crop_size, dataset.crop_size, 1])  # 注意图片必须是三维的
    dens_true = tf.squeeze(tf.slice(dens_true, (2, 0, 0, 0), (1, -1, -1, -1)), axis=0)
    const_tensor = tf.ones((dataset.crop_size, dataset.crop_size, 1))
    '''
    for index in range(5):
        temp_pic = pic_true.numpy()
        temp_dens = np.squeeze(dens_true.numpy())
        plt.imshow(temp_pic[index])
        plt.show()
        plt.imshow(temp_dens[index])
        plt.show()
    '''
    return pic_true, dens_true, const_tensor


def spatial_net():
    input_data = keras.Input(shape=(160, 160, 3))
    digits = input_data
    digits = KL.Conv2D(20, (5, 5), activation='relu',  padding='same')(digits)
    digits = KL.MaxPool2D((2, 2), padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.MaxPool2D((2, 2), padding='same')(digits)

    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)

    digits = KL.Conv2D(20, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(10, (3, 3), activation='relu',  padding='same')(digits)

    digits = KL.Conv2D(1, (1, 1), activation='relu',  padding='same')(digits)

    digits = KL.UpSampling2D(size=(4, 4))(digits)
    prediction = digits
    s_net = keras.Model(inputs=input_data, outputs=prediction)
    return s_net


def temporal_net():
    input_data = keras.Input(shape=(5, 160, 160, 3))
    digits = input_data
    digits = KL.Conv3D(20, (3, 3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv3D(40, (3, 3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv3D(20, (3, 3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv3D(1, (1, 1, 1), activation='relu',  padding='same')(digits)
    prediction = digits
    t_net = keras.Model(inputs=input_data, outputs=prediction)
    return t_net


def fusion_net():
    s_input_data = keras.Input(shape=(160, 160, 1))
    t_input_data = keras.Input(shape=(160, 160, 1))
    input_const = keras.Input(shape=(160, 160, 1))
    digits = KL.Concatenate(axis=-1)([s_input_data, t_input_data])
    digits = KL.Conv2D(10, (5, 5), padding='same')(digits)
    digits = KL.Conv2D(20, (3, 3), padding='same')(digits)
    digits = KL.Conv2D(10, (3, 3), padding='same')(digits)
    digits = KL.Conv2D(1, (1, 1), padding='same')(digits)

    s_attention = KL.Activation(activation='sigmoid')(digits)
    t_attention = KL.Subtract()([input_const, s_attention])
    s_prediction = KL.Multiply()([s_input_data, s_attention])
    t_prediction = KL.Multiply()([t_input_data, t_attention])
    prediction = KL.Add()([s_prediction, t_prediction])
    fus_net = keras.Model(inputs=[s_input_data, t_input_data, input_const], outputs=prediction)
    return fus_net


def slice_layer(input_data):
    sliced_data = tf.slice(input_data, (0, 2, 0, 0, 0), (-1, 1, -1, -1, -1))
    squeezed_data = tf.squeeze(sliced_data, axis=1)
    return squeezed_data


def final_model():
    input_data = keras.Input(shape=(5, 160, 160, 3))
    input_const = keras.Input(shape=(160, 160, 1))
    s_net = spatial_net()
    t_net = temporal_net()
    fus_net = fusion_net()

    s_input_data = input_data
    s_input_data = KL.Lambda(slice_layer)(s_input_data)
    t_input_data = input_data
    s_output = s_net(s_input_data)
    t_output = t_net(t_input_data)
    t_output = KL.Lambda(slice_layer)(t_output)

    fus_output = fus_net([s_output, t_output, input_const])
    prediction = fus_output
    f_model = keras.Model(inputs=[input_data, input_const], outputs=prediction)
    return f_model


def euclidean_distance_loss(y_true, y_pred):
    loss_1 = keras.losses.mean_squared_error(y_true, y_pred)  # 注意对图片来说，loss针对的是图中的每一个像素点
    loss_2 = tf.sqrt(tf.reduce_sum(loss_1, axis=[1, 2]))
    loss_3 = tf.reduce_mean(loss_2, axis=0)
    return loss_3


def save_model(model: keras.Model, json_path):
    model_json_data = model.to_json()
    with open(json_path, 'w') as json_file:
        json_file.write(model_json_data)


def load_model(model_p, weight_p):
    json_file = open(model_p, 'r')
    medel_json_data = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(medel_json_data)
    loaded_model.load_weights(weight_p)
    return loaded_model


if __name__ == "__main__":
    tfrecord_path = os.path.join(dataset.set_root, dataset.set_name, 'train.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(parse_image_function)
    processed_dataset = parsed_dataset.map(process_function)
    batched_dataset = processed_dataset.batch(BATCHSIZE)
    temp_net = final_model()
    save_model(temp_net, model_path+'model.json')
    for epoch in range(1000):
        for index, data in enumerate(batched_dataset):
            # for repeat in range(20):
            with tf.GradientTape() as train_tape:
                opti = tf.train.AdamOptimizer(learning_rate=1e-5)
                predict = temp_net([data[0], data[2]], training=True)  # 注意所有的keras模型必须添上一句话，training=True
                loss = euclidean_distance_loss(data[1], predict)
                gradiens = train_tape.gradient(loss, temp_net.variables)
                opti.apply_gradients(zip(gradiens, temp_net.variables))
                print(epoch, index, np.sum(loss.numpy()))
                if index % 100 == 0:
                    plt.imshow(np.squeeze(data[1][0]))
                    plt.savefig(model_path+'temp_fig/epoch_%s_index_%s_dens_true.jpg' % (epoch, index))
                    plt.imshow(np.squeeze(predict[0]))
                    plt.savefig(model_path+'temp_fig/epoch_%s_index_%s_dens_pre.jpg' % (epoch, index))
                    plt.imshow(np.squeeze(data[0][0][2]).astype(np.int))
                    plt.savefig(model_path+'temp_fig/epoch_%s_index_%s_true.jpg' % (epoch, index))
        temp_net.save_weights(model_path+'weight_epoch_%s.h5' % epoch)

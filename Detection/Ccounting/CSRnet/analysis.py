import tensorflow as tf
from tensorflow import keras
import process
import model
import os
import numpy as np
from matplotlib import pyplot as plt


def load_model(model_p, weight_p):
    json_file = open(model_p, 'r')
    medel_json_data = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(medel_json_data)
    loaded_model.load_weights(weight_p)
    return loaded_model


def summary_numpy(scatted_np: np.array):
    scatted_np = scatted_np.squeeze()
    merge_left = np.concatenate((scatted_np[0], scatted_np[1]), axis=0)
    merge_right = np.concatenate((scatted_np[2], scatted_np[3]), axis=0)
    result = np.concatenate((merge_left, merge_right), axis=1)
    return result


def show(img_array):
    temp_array = img_array*255.0
    temp_array = temp_array.astype(np.uint8)
    plt.imshow(temp_array)
    plt.show()


def show_errors():
    pass


if __name__ == '__main__':
    model_path = 'Datasets/shtech/model.json'
    weight_path = 'Datasets/shtech/weight_226000.h5'
    mynet = load_model(model_path, weight_path)
    shtech_image_path, shtech_set_path = process.get_shtech_path()
    tfrecord_path = os.path.join(shtech_set_path[0][1], 'all_data.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(model.parse_image_function)
    processed_dataset = parsed_dataset.map(model.process_function)
    batched_dataset = processed_dataset.batch(9)
    for dataset in batched_dataset:
        imgs_tensor = dataset[0][:4]
        images = dataset[0][:4].numpy()
        predic = mynet(imgs_tensor).numpy()
        truth = dataset[1][:4].numpy()
        sum_predic = summary_numpy(predic)
        sum_truth = summary_numpy(truth)
        sum_images = summary_numpy(images)
        show(sum_images)
        show(sum_predic)
        show(sum_truth)
    pass

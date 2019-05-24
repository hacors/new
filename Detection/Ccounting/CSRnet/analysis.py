import tensorflow as tf
from tensorflow import keras


def load_model():
    json_file = open('Datasets/shtech/model.json', 'r')
    medel_json_data = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(medel_json_data)
    loaded_model.load_weights('Datasets/shtech/weight.h5')
    return loaded_model


if __name__ == '__main__':
    mynet = load_model()

    print(mynet.summary())
    pass

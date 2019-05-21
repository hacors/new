from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
# VGG-16 instance
model = VGG16(weights='imagenet', include_top=True)
the_image = Image('Temp/images/golden.jpg')
# image = load_img('Temp/images/golden.jpg', target_size=(224, 224))
# the_image = img_to_array(image)
# reshape it into the specific format
# the_image = the_image.reshape((1,) + the_image.shape)
# print(the_image.shape)
# prepare the image data for VGG
the_image = preprocess_input(the_image)
# using the pre-trained model to predict
prediction = model.predict(the_image)
# decode the prediction results
results = decode_predictions(prediction, top=3)
print(results)

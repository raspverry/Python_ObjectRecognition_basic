import tensorflow as tf
import numpy as np
from PIL import Image


filename = './Images/dog.jpg'
im = Image.open(filename)
im = im.resize((224,224))
#im.show()

# deep learning module weight - pre-trained
#mobile = tf.keras.applications.mobilenet.MobileNet()  #v1
mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()

imgArray = np.array(im)
final_image = np.expand_dims(imgArray, axis =0) ## need fourth dimension
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)
print(final_image.shape)
prediction = mobile.predict(final_image)
result = tf.keras.applications.imagenet_utils.decode_predictions(prediction)
print(result)



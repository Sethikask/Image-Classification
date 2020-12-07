# Import modules
import numpy as np
from tensorflow.keras.applications import (vgg16)
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import urllib
import os

# Initialize the models
vgg_model = vgg16.VGG16(weights='imagenet')

def processImage(filename):
  original = load_img(filename, target_size=(224, 224))
  numpy_image = img_to_array(original)
  image_batch = np.expand_dims(numpy_image, axis=0)
  processed_image = vgg16.preprocess_input(image_batch.copy())
  predictions = vgg_model.predict(processed_image)
  label_vgg = decode_predictions(predictions)
  for prediction_id in range(len(label_vgg[0])):
      print(label_vgg[0][prediction_id])

os.system("clear")

while True:
  filename = input(">>")
  if filename == "quit":
    break
  elif filename != "":
    print("")
    try:
        urllib.request.urlretrieve(filename, "image.jpg")
        processImage("image.jpg")
        print("")
        print("")
    except Exception as e:
        print(e)

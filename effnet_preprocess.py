import cv2
import tensorflow as tf
import numpy as np


def preprocess_ig(image):
  image = cv2.resize(image, (224, 224))
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = tf.convert_to_tensor(image, dtype=tf.float32)
  return image

def preprocess_saliency(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)
    image = image.reshape(1,224,224,3)
    return image

def gradcam_preprocess(image):
  image = cv2.resize(image, (224, 224))
  image = image.astype('float32') 
  image = np.expand_dims(image, axis=0)
  return image

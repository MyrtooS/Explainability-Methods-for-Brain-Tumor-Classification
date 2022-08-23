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
  image = np.expand_dims(image, axis=0)
  image = tf.keras.applications.vgg16.preprocess_input(image, data_format=None)
  image = image.reshape(1,224,224,3)
  return image

def gradcam_preprocess(image):
  image = cv2.resize(image, (224, 224))
  image = np.expand_dims(image, axis=0)
  image = tf.keras.applications.vgg16.preprocess_input(image, data_format=None)
  return image
import numpy as np
import cv2
import tensorflow as tf


def saliency_preproccess(image,model):
  images = tf.Variable(image, dtype=float)

#calculate the gradient with respect to the top class score
  with tf.GradientTape() as tape:
      pred = model(images, training=False)
      class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
      loss = pred[0][class_idxs_sorted[0]]
    
  grads = tape.gradient(loss, images)
  return grads

def make_saliency(grads):
  dgrad_abs = tf.math.abs(grads)
#find the max of the absolute values of the gradient along each RGB channel
  dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

## normalize to range between 0 and 1
  arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
  grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
  return grad_eval

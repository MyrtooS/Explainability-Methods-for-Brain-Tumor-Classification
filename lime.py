import cv2
import numpy as np
import skimage.io 
import skimage.segmentation
from keras.applications.imagenet_utils import decode_predictions
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings


def lime(Xi,y_model):
  Xi = cv2.resize(Xi,(224,224))
  Xi = Xi.astype('double')

  superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
  num_superpixels = np.unique(superpixels).shape[0]

  num_perturb = 68
  perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

  predictions = []
  for pert in perturbations:
    perturbed_img = perturb_image(Xi,pert,superpixels)
    pred = y_model.predict(perturbed_img[np.newaxis,:,:,:])
    predictions.append(pred)

  predictions = np.array(predictions)
  predictions.shape

  original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
  distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
  # distances.shape
  #Kernel function
  kernel_width = 0.25
  weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2))
  # weights.shape

  np.random.seed(222)
  preds = y_model.predict(Xi[np.newaxis,:,:,:])\

  top_pred_classes = preds[0].argsort()[-5:][::-1]
  # top_pred_classes  

  class_to_explain = top_pred_classes[0]
  simpler_model = LinearRegression()
  simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
  coeff = simpler_model.coef_[0]
  # coeff

  num_top_features = 4
  top_features = np.argsort(coeff)[-num_top_features:] 
# top_features
  return top_features, num_superpixels, superpixels, Xi


def perturb_image(img,perturbation,segments):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1 
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image
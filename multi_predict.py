import cv2
import numpy as np

#this method predicts the class of an image for the multi classification
def predict_image_multi(image,model):

  ypred = model.predict(image) 
  j= np.argmax(ypred[0])

  if j == 0:
    print('pituitary tumor')
  elif j == 1:
    print('glioma tumor')
  elif j == 2:
    print('meningioma tumor')
  elif j == 3:
    print('no tumor')
  else:
    print('Something went wrong')
  print(ypred)
  print(j)
  return j, image, model, ypred

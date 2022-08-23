from tf_explain.core.smoothgrad import SmoothGrad
import cv2
import matplotlib.pyplot as plt


def smoothgrad(image, class_index, model, noise):
  image = cv2.resize(image,(224, 224))
  data = ([image], None)

  class_index = 2
  explainer = SmoothGrad()
# Compute SmoothGrad on VGG16
  grid = explainer.explain(data, model, class_index, 20, noise)
  explainer.save(grid, ".", "smoothgrad.png")
  fig, ax = plt.subplots(1,2,figsize=(10,10))

  ax[0].imshow(grid)
  ax[1].imshow(image)
  return grid
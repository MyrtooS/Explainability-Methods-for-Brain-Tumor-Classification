from re import I
import numpy as np
import h5py
import os
from PIL import Image
import glob
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import expand_dims
import cv2

path = "/home/myrto/mri_kaggle/Training/"
image_data = []
IMG_SIZE = 300


classes = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]
for c in classes:
        im= []
        for filename in os.listdir(os.path.join(path,c)):
                
                path2 = os.path.join((os.path.join(path,c)),filename)
                image = Image.open(path2)
                image = image.resize((300,300))
                image = np.array(image)
                im.append(np.array(image))
                img = np.array(im)
    
        datagen = ImageDataGenerator(height_shift_range=0.2, width_shift_range=0.2)
        PREFIX = 'Shifted'
        imGen = datagen.flow(img, batch_size=1, save_to_dir = (os.path.join(path,c)), 
                save_prefix=PREFIX, save_format='jpg')
        for batch in range(100):
                batch = imGen.next()

   

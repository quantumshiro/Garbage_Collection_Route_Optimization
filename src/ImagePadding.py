from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

N_img = 1000

input_path = 'data/image/ok'
files = glob.glob(input_path + '/*.jpg')

ouput_path = 'data/image/ok'
if os.path.isdir(ouput_path) == False:
    os.mkdir(ouput_path)
    
for i, file in enumerate(files):
    img = load_img(file)
    x   = img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    
    # Generate ImageDataGenerator
    datagen = ImageDataGenerator (
        zca_epsilon=1e-06,   
        rotation_range=10.0,
        width_shift_range=0.0, 
        height_shift_range=0.0, 
        brightness_range=None, 
        zoom_range=0.0,       
        horizontal_flip=True, 
        vertical_flip=True, 
    )
    
    dg = datagen.flow(x, batch_siz=1, save_to_dir=ouput_path, save_prefix='img', save_format='jpg')
    for i in range(170, N_img):
        batch = dg.next()
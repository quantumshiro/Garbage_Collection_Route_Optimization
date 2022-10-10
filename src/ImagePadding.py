from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob

# 1枚あたり20枚の画像を水増し
N_img = 30

# 入力画像の保存先パス
input_path = 'data/image/ok'
files = glob.glob(input_path + '/*.jpg')
 
# 出力画像の保存先パス
output_path = "data/image/ok"
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
 
 
for i, file in enumerate(files):
 
    img = load_img(file)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
 
    # ImageDataGeneratorの生成
    datagen = ImageDataGenerator(
        zca_epsilon=1e-06,   # 白色化のイプシロン
        rotation_range=10.0, # ランダムに回転させる範囲
        width_shift_range=0.1, # ランダムに幅をシフトさせる範囲
        height_shift_range=0.1, # ランダムに高さをシフトさせる範囲
        brightness_range=None, # ランダムに明るさを変化させる範囲
        zoom_range=0.0,        # ランダムにズームさせる範囲
        horizontal_flip=True, # ランダムに水平方向に反転させる
        vertical_flip=True, # ランダムに垂直方向に反転させる
    )
 
    # 1枚あたり20枚の画像を水増し生成
    dg = datagen.flow(x, batch_size=1, save_to_dir=output_path, save_prefix='', save_format='jpg')
    for i in range(N_img):
        batch = dg.next()


# 001003.jpg, 00104.jpg ...
tmp_dir = 'data/image/ok'
for i, file in enumerate(os.listdir(tmp_dir)):
    os.rename(tmp_dir + '/' + file, tmp_dir + '/' + str(i+1).zfill(6) + '.jpg')
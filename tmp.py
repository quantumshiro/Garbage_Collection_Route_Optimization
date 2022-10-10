import glob
import os

tmp_dir = 'data/image/ok'
for i, file in enumerate(os.listdir(tmp_dir)):
    os.rename(tmp_dir + '/' + file, tmp_dir + '/' + str(i+1).zfill(6) + '.jpg')
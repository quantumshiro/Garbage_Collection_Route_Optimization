import os
import glob

tmp_dir = 'data/image/ngs'
for i, file in enumerate(os.listdir(tmp_dir)):
    os.rename(tmp_dir + '/' + file, tmp_dir + '/' + str(i+1).zfill(6) + '.jpg')

# encoding=utf-8
import os
import numpy as np
import cv2
import natsort

from .LabStretching import LABStretching
from .global_stretching_RGB import stretching
from .relativeglobalhistogramstretching import RelativeGHstretching

np.seterr(over='ignore')
if __name__ == '__main__':
    pass


folder = "C:/Users/shamil/Desktop"
path = folder + "/InputImages"
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        img = cv2.imread(folder +'/InputImages/' + file)

        height = len(img)

        width = len(img[0])

        sceneRadiance = img

        sceneRadiance = stretching(sceneRadiance)

        sceneRadiance = LABStretching(sceneRadiance)

        cv2.imwrite(folder+'/OutputImages/' + prefix + '_RGHS.jpg', sceneRadiance)

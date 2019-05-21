import sys
sys.path.insert(0, '..')

import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from load_model import load_flattened_model, load_model
from predict import PredictWithRotation
from canvas import canvas_with_skeleton

# model = load_flattened_model()
model = load_model()

img = cv2.imread('media/cr7_2.jpg')
keypoints = PredictWithRotation.predict(model, img, 90)
# keypoints = keypoints_selection(keypoints_list)
canvas = canvas_with_skeleton(img, keypoints)
plt.imshow(canvas)
# canvas = canvas_with_skeleton(img, keypoints_list, 270)
# cv2.imwrite('canvas.jpg', canvas)

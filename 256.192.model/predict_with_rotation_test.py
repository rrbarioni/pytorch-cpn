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

input_img = cv2.imread('media/cr7_2.jpg')
keypoints = PredictWithRotation.predict(model, input_img, 10)
canvas = canvas_with_skeleton(input_img, keypoints)
plt.imshow(canvas)
# cv2.imwrite('canvas.jpg', canvas)

keypoints_list = PredictWithRotation.predict(model, input_img, 10)
for i, r in enumerate(range(0, 360, 10)):
    cv2.imwrite('media/canvas%s.jpg' % r, canvas_with_skeleton(input_img, keypoints_list[i]))
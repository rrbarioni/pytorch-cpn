import sys
sys.path.insert(0, '..')

import time
import cv2
import numpy as np

from load_model import load_flattened_model, load_model
from predict import Predict
from canvas import canvas_with_skeleton

# model = load_flattened_model()
model = load_model()

cap = cv2.VideoCapture(0)
while(True):
    _, img = cap.read()

    t = time.time()
    keypoints = Predict.predict(model, img)
    t = time.time() - t
    
    canvas = canvas_with_skeleton(img, keypoints)
    canvas = cv2.resize(canvas, None, fx=2, fy=2)
    cv2.putText(canvas, 'fps: %s' % (1/t), (60, 60), cv2.FONT_HERSHEY_SIMPLEX,
        2, 255)
    cv2.imshow('frame', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

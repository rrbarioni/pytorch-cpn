import math
import cv2
import numpy as np

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170],
    [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

keypoints_pairs = [(0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), (6,8),
    (7,9), (8,10), (0,11), (0,12), (11,13), (12,14), (13,15), (14,16)]

def canvas_with_skeleton(canvas, keypoints, rotation=None):
    if rotation is not None:
        keypoints = [(x, y) for (p, r, x, y, c) in keypoints if r == rotation]
    for i in range(len(keypoints_pairs)):
        cur_canvas = canvas.copy()
        
        ki1, ki2 = keypoints_pairs[i]
        y1, x1 = keypoints[ki1]
        y2, x2 = keypoints[ki2]
        X = np.array([x1, x2])
        Y = np.array([y1, y2])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length / 2), 4),
            int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
    for i in range(len(keypoints)):
        cv2.circle(canvas, keypoints[i], 4, colors[i], thickness=-1)

    return canvas
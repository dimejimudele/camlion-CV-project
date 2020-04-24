#!/usr/bin/env python

import argparse
import cv2
import numpy as np

class BirdEyeView(object):
    def __init__(self, capture):
        self.capture = capture
        self.image_coordinates = []

    def get_four_point_view(self):
        if self.capture.isOpened():
            (self.status, self.frame) = self.capture.read()
            cv2.imshow('first frame', self.frame)

            self.clone = self.frame.copy()
            cv2.namedWindow('first frame')
            cv2.setMouseCallback('first frame', self.extract_coordinates)

            key = cv2.waitKey(0) & 0xFF
            if (key == ord('q')):
                cv2.destroyAllWindows()
                exit(1)
            else:
                cv2.destroyWindow('first frame')

    def extract_coordinates(self, event, x, y, flags, parameters):
        if (event != cv2.EVENT_LBUTTONDOWN):
            return

        if (len(self.image_coordinates) == 4):
            print('Maximum numbers of coordinates entried. Please click any key to continue.')
            return

        radius = 3
        thickness = 4
        color = (0,0,255) # Red
        cv2.circle(self.clone, (x,y), radius, color, thickness)
        self.image_coordinates.append((x,y))

        # Draw rectangle around ROI
        #cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)
        cv2.imshow('first frame', self.clone)

    def ordered_points(self):
        pts = np.array(self.image_coordinates, dtype = "int32")
        rect = np.zeros((4, 2), dtype = "int32")

        sum = pts.sum(axis = 1)
        top_left = np.argmin(sum)
        bottom_right = np.argmax(sum)
        rect[0] = pts[top_left] # top-left
        rect[3] = pts[bottom_right] # bottom-right

        pts = np.delete(pts, top_left, 0)
        bottom_right = bottom_right if (bottom_right < top_left) else (bottom_right - 1)
        pts = np.delete(pts, bottom_right, 0)

        diff = np.diff(pts, axis = 1)
        rect[2] = pts[np.argmin(diff)] # top-right
        rect[1] = pts[np.argmax(diff)] # bottom-left

        return rect

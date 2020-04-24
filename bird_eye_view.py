#!/usr/bin/env python

import argparse
import cv2
from pynput.keyboard import Key

class BirdEyeView(object):
    def __init__(self, capture):
        self.capture = capture
        self.image_coordinates = []

    def get4pointsView(self):
        print(self.capture.isOpened())
        if self.capture.isOpened():
            (self.status, self.frame) = self.capture.read()
            cv2.imshow('first frame', self.frame)

            self.clone = self.frame.copy()
            cv2.namedWindow('first frame')
            cv2.setMouseCallback('first frame', self.extract_coordinates)

            key = cv2.waitKey(0)
            if (key == ord('q')):
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                exit(1)
            else:
                print(self.image_coordinates)
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

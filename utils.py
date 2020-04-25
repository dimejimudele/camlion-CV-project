from enum import Enum
import numpy as np
import cv2
import math

class PersonDetection:

    def __init__(self, centroid, width, height):
        self.centroid = centroid    # tuple (x,y) [pixeis]
        self.width = width          # bounding_box width       
        self.height = height        # bounding_box height
        self.bottom_point = self.compute_bottom_point() #bottom points is in source image pixel coordinates

        # person is safe until someone is detected nearby
        self.safe = True

        self.coordinates = None

    def compute_bottom_point(self):
        return (self.centroid[0], self.centroid[1] + self.height/2)


class Homography:

    def __init__(self):
        pass

    def transform_point(self, src_point):
        pt1 = np.array([src_point[0], src_point[1], 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.H, pt1)
        pt2 = pt2/pt2[2]
        trg_point = (int(pt2[0]), int(pt2[1]))
        return trg_point

    def compute_homography(self, src_pts, trg_pts):
        self.H, _ = cv2.findHomography(np.array(src_pts), np.array(trg_pts))


def euclidian_distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)



def interpolation(x1, y1, x2, y2, x):
    return y1 + ((x-x1)/(x2-x1))*(y2-y1)


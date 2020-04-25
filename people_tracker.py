#!/usr/bin/env python

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import math
from utils import *
from tkinter import *
from bird_eye_view import BirdEyeView


def people_tracker(args):

    # load the COCO class labels our YOLO model was trained on
    labelsPath = args["labels"]
    LABELS = open(labelsPath).read().strip().split("\n")

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromDarknet(args["config"], args["model"])

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # if a video path was not supplied, grab a reference to the webcam
    if not args.get("input", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(args["input"])

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # Setup bird eye view
    start = False
    while (not start):
        print("Please input coordenates in image.")
        birdEyeView = BirdEyeView(vs)
        birdEyeView.get_four_point_view()
        start = continue_after_bird_eye()

    print('******************************************')
    # [[top-left], [bottom-left], [top-right], [bottom-right]]
    plane_coordinates = birdEyeView.ordered_points()
    print_plane_coordinates(plane_coordinates)

    # start the frames per second throughput estimator
    fps = FPS().start()


    #use the fisrt frame for initial settings
    first_frame = vs.read()
    first_frame = first_frame[1] if args.get("input", False) else first_frame
    first_frame = imutils.resize(first_frame, width=1000)

    src_image_width = first_frame.shape[1]
    src_image_height = first_frame.shape[0]


    #syntatic data until Clarissa code is added
    #rectangule is hard coded for now
    pts_src = [[606, 106], [806, 132], [511, 434],[201, 354]]
    cv2.circle(first_frame, tuple(pts_src[0]), 4, (0, 0, 0), -1)
    cv2.circle(first_frame, tuple(pts_src[1]), 4, (0, 0, 0), -1)
    cv2.circle(first_frame, tuple(pts_src[2]), 4, (0, 0, 0), -1)
    cv2.circle(first_frame, tuple(pts_src[3]), 4, (0, 0, 0), -1)

    #target side points
    # pts_dst = np.array([[int(src_image_width/8),0], [int(src_image_width*3/8),0], [int(src_image_width*3/8),src_image_height], [int(src_image_width/8),src_image_height]])
    pts_dst = [[int(440),250], [int(500),250], [int(500),350], [int(440),350]]

    # Calculate Homography
    homography = Homography()
    homography.compute_homography(pts_src, pts_dst)

    #display distorted image
    im_out = cv2.warpPerspective(first_frame, homography.H, (first_frame.shape[1], first_frame.shape[0]))
    cv2.imshow("Destination Image", im_out)

    #transform the 4 corners
    transformedCorners = []
    transformedCorners.append(homography.transform_point((0, 0)))
    transformedCorners.append(homography.transform_point((src_image_width, 0)))
    transformedCorners.append(homography.transform_point((src_image_width, src_image_height)))
    transformedCorners.append(homography.transform_point((0, src_image_height)))

    transformered_height = transformedCorners[2][1] - transformedCorners[1][1]
    transformered_width = transformedCorners[2][0] - transformedCorners[0][0]

    #Initiates Map
    tk = Tk()
    canvas = Canvas(tk, width=500, height=500, bd=0, highlightthickness=0)
    canvas.pack()

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame


        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        frame = imutils.resize(frame, width=1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= bightness_contrast_enhance(frame, alpha = args["alpha"], beta = args["beta"])

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if (totalFrames % args["skip_frames"] != 0):
            totalFrames += 1
            fps.update()
            continue

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        centroids = []
        people_detected = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if LABELS[classID] == "person":
                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            centroids.append((centerX, centerY))

        my_idx = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        # ensure at least one detection exists
        if len(my_idx) > 0:

            # loop over the indexes we are keeping
            for i in my_idx.flatten():

                # extract with and height of bounding box
                (w, h) = (boxes[i][2], boxes[i][3])

                #create PersonDetection if bounding box is validated
                people_detected.append(PersonDetection(centroids[i], w, h))


        for i in range(len(people_detected)):

            transformed_bottom_point = homography.transform_point(people_detected[i].bottom_point)

            people_detected[i].coordinates = (interpolation(transformedCorners[0][0],0,transformered_width +  transformedCorners[0][0], 400,transformed_bottom_point[0]),
                                    interpolation(transformedCorners[1][1],0,transformered_height +  transformedCorners[1][1], 400,transformed_bottom_point[1]))

            #iterates over every person again
            for j in range (i+1, len(people_detected)):
                transformed_point = homography.transform_point(people_detected[j].bottom_point)

                pixel_distance = euclidian_distance(transformed_bottom_point[0], transformed_bottom_point[1], transformed_point[0], transformed_point[1])

                #hardcoded for now
                distance = 5.5*pixel_distance/60

                if distance < 2:
                    people_detected[i].safe = False
                    people_detected[j].safe = False



        #representation
        for person in people_detected:
            # draw a bounding box rectangle and label on the frame

            #if person is safe color is green otherwise is red
            color = (0, 255, 0) if person.safe else (0, 0, 255)

            #compute bounding box coordinates
            bb_x = int(person.centroid[0] - person.width/2)
            bb_y = int(person.centroid[1] - person.height/2)

            cv2.rectangle(frame, (bb_x, bb_y), (bb_x + person.width, bb_y + person.height), color, 2)

            color = "green" if person.safe else "red"
            canvas.create_oval(person.coordinates[0] - 5, person.coordinates[1] - 5, person.coordinates[0] + 5, person.coordinates[1] + 5, fill=color)

        #number of people
        print("PEOPLE IN IMAGE: {}".format(len(people_detected)))

        canvas.update_idletasks()
        canvas.update()

        # show the output frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

        canvas.delete("all")

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if not args.get("input", False):
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()

def bightness_contrast_enhance(image, alpha = 1.6, beta = 0):

    """
        alpha  = Contrast control (1.0-3.0)
        beta = # Brightness control (0-100)

    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def clahe_denoising_enhancement(image, tile_grid_size = (8, 8), clip_limit = 0.3):
    """
    image: RGB

    """
    lab_frame = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab_frame)
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    lab_frame_enhanced = cv2.merge((cl,a,b))
    #-----Converting image from LAB Color model to RGB model--------------------
    return cv2.cvtColor(lab_frame_enhanced, cv2.COLOR_LAB2RGB)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", required=True,
        help="path to the yolo config file.")
    parser.add_argument("-m", "--model", required=True,
        help="path to the yolo weight file")
    parser.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    parser.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    parser.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    parser.add_argument("-s", "--skip_frames", type=int, default=3,
        help="# of skip frames between detections")
    parser.add_argument("-l", "--labels", type=str, help="path to the classes file")
    parser.add_argument("-t", "--threshold", type=float, default=0.2,
	   help="threshold when applying non-maxima suppression")
    parser.add_argument("-a", "--alpha", type=float, default=1.5,
        help="alpha parameter for for input frame contrast enhancement (0 - 3")
    parser.add_argument("-b", "--beta", type=float, default= 0,
        help="beta parameter for input frame brightness enhancement (0 - 100)")
    args = vars(parser.parse_args())

    people_tracker(args)

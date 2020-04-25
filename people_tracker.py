# import the necessary packages
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

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

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

    # start the frames per second throughput estimator
    fps = FPS().start()

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

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame= bightness_contrast_enhance(frame, alpha = args["alpha"], beta = args["beta"])
      

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

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
        classIDs = []
        centroids = []

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
                            centroids.append((int(centerX), int(centerY)))
                            classIDs.append(classID)
        
        my_idx = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        # ensure at least one detection exists
        if len(my_idx) > 0:
            # loop over the indexes we are keeping
            for i in my_idx.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #draw the person centroid
                cv2.circle(frame, centroids[i], 4, color, -1)

                if i != 0:
                    cv2.line(frame, centroids[i], centroids[i-1], (0, 255, 255), 2)
        
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()
        
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
    parser.add_argument("-s", "--skip_frames", type=int, default=2,
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



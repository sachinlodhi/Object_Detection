########### REAL TIME OBJECT DETECTION #########################

#########################################################################
#   Author : Sachin Lodhi                                               #
#   Track  : Computer Vision and Internet of Things                     #
#   Batch  : June 2021                                                  #
#   Assignment 1 : REAL TIME OBJECT DETECTION                           #
#   Graduate Rotational Internship Program,                             #
#   The Spark Foundation, June 2021                                     #
#                                                                       #
#########################################################################

import objDetectModule as odm
import cv2 as cv
import argparse

# Argument Parsing section for flexibility of
parser = argparse.ArgumentParser()

parser.add_argument("-vs", "--vidSource", required=True, choices=["0", "1"],
                    help = "Source of the video feed i.e. camera number . Use 0 for in-built camera or 1 for Secondary Camera")
args = vars(parser.parse_args())


cap = cv.VideoCapture(int(args["vidSource"]))
objDetector = odm.objDetection()
while True :
        _, img = cap.read()
        processedFrame = objDetector.locateObj(img)

        cv.imshow("Live Object Detection", processedFrame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

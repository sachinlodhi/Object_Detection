import objDetectModule as odm
import cv2 as cv

cap = cv.VideoCapture(0)
objDetector = odm.objDetection()
while True :
        _, img = cap.read()
        processedFrame = objDetector.locateObj(img)

        cv.imshow("output", processedFrame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

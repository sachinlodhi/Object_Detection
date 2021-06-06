# This is the custom module to use with other files

import cv2 as cv

class objDetection():
    def __init__(self ):

        self.classLabels = []
        self.classFile = './weights/coco.names'

        with open(self.classFile, 'rt') as f:
            self.classLabels = f.read().rstrip('\n').split('\n')

        self.weightsPath = './weights/frozen_inference_graph.pb'
        self.configPath = './weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

        self.net = cv.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
    def locateObj(self, frame, conf=0.5):
        classIds, confidences, bbox = self.net.detect(frame, confThreshold=conf)
        try:
            for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bbox):
                cv.rectangle(frame, box, color=(0, 0, 255), thickness=2)
                cv.putText(frame, self.classLabels[classId - 1], (box[0] + 10, box[1] + 30),
                           cv.FONT_ITALIC, 1, (0, 0, 255), 2)
            return frame
        except:
            return False
            pass



def main():
    cap = cv.VideoCapture(0)

    objDetector = objDetection()
    while True :
        _, img = cap.read()
        processedFrame = objDetector.locateObj(img)

        cv.imshow("output", processedFrame)


        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

import cv2
import numpy as np

cap = cv2.VideoCapture(0) #capture video from camera
whT = 320 # height and width parameters of the image
confThreshold = 0.5
nmsThreshold = 0.3

## Loading model

#load category names
classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#constructing the net, first import config and weights
#yolov3.weights and config is not included in my repository
#it can be downloaded from YOLO website
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def FindObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIDs = []
    confs = []

    #search for best confidences for particular class of object in the box
    for output in outputs:
        for detection in output:
            #first 5 columns do not include categories so we exclude them
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                w, h = int(detection[2]*wT) , int(detection[3]*hT)
                x, y = (int((detection[0]*wT) - w/2)) , (int((detection[1]*hT) - h/2))
                bbox.append([x,y,w,h])
                classIDs.append(classID)
                confs.append(float(confidence))
    
    #no maximal suppresion - so we have no multiple boxes identifying same object
    indicies = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    #making frames with names and accuracies
    for i in indicies:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (65,105,225), 2)
        cv2.putText(img, f'{classNames[classIDs[i]].upper()} {int(confs[i]*100)}%',
                    (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65,105,225), 2)

#loop takes the captured video, forwards it into the net and extracts outputs
while True:
    success, img = cap.read()

    #we create blob from our image
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1 ,crop = False)
    net.setInput(blob)

    outputNames = list(net.getUnconnectedOutLayersNames())
    outputs = list(net.forward(outputNames))
    
    #output columns: cx, cy, w, h, bbox confidence, confidences for particular class of object in the box
    
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    FindObjects(outputs, img)

    cv2.imshow('YOLO-opencv', img)
    cv2.waitKey(1)

# -*- coding: utf-8 -*-


from tensorflow.python.ops.gen_math_ops import arg_max
from MyFunctions import sentiment_from_faces
from tensorflow.keras.models import load_model
import cv2
import imutils
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
classifier =load_model('emotion_detection.h5')
classifier.load_weights('model_weight.h5')

deployPrototxt = "deploy.prototxt"
deployPrototxt_weight = "res10_300x300_ssd_iter_140000.caffemodel"
face_detection_model = cv2.dnn.readNet(deployPrototxt,deployPrototxt_weight)

frame = cv2.imread("disguest.jfif")

image = frame.copy()

#detect faces in the frame and preict if their sentiment
    #extract coordination of faces
(coordination,sentiment_face_detection)=sentiment_from_faces(image,face_detection_model,classifier)
for (box,prediction) in zip(coordination,sentiment_face_detection):
    (start_x,start_y,end_x,end_y)=box
    sentiment=np.argmax(prediction)
    resultat= ["angry","disgust","fear","happy","neutral","sad","surprise"]

        #determine the class label and color we will use to draw the bounding box and text
    label=resultat[sentiment]
    color=(0,255,0)

        #display the label and bounding boxes
    cv2.putText(image,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
    cv2.rectangle(image,(start_x,start_y),(end_x,end_y),color,2)

cv2.imshow("OutPut.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

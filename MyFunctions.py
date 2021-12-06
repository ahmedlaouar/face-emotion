import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

def sentiment_from_faces(frame, face_model, mask_detect_model):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    
    face_model.setInput(blob)
    detections = face_model.forward()

    faces = []
    coordinations = []
    predictions = []
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
        if confidence>0.5:
            #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
            
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))

            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face=cv2.resize(face,(48,48))
            face=img_to_array(face)
            face=preprocess_input(face)
            faces.append(face)
            coordinations.append((startX, startY, endX, endY))
    faces2 = np.array(faces, dtype="float32")
    predictions = mask_detect_model.predict(faces2, batch_size=32)
    return (coordinations, predictions)


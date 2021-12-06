#!/usr/bin/env python
# coding: utf-8

# In[1]:


from MyFunctions import sentiment_from_faces
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# # loading face detector model

# In[2]:


deployPrototxt = "deploy.prototxt"
deployPrototxt_weight = "res10_300x300_ssd_iter_140000.caffemodel"
face_detection_model = cv2.dnn.readNet(deployPrototxt,deployPrototxt_weight)


# # load face mask detector

# In[3]:


classifier =load_model('emotion_detection.h5')
classifier.load_weights('model_weight.h5')


# In[7]:


video_stream=VideoStream(src=0).start()

while True:
    #grab the frame from the threaded video stream and
    frame=video_stream.read()
    #resize the frame to have a maximum width of 400 pixels
    frame=imutils.resize(frame,width=400)


    try:
        #detect faces in the frame and preict if they are waring masks or not
        #extract coordination of faces
        (coordination,sentiment_face_detection)=sentiment_from_faces(frame,face_detection_model,classifier)
        resultat= ["angry","disgust","fear","happy","neutral","sad","surprise"]

    #loop over the detected face and their corrosponding coordinations
        for (box,prediction) in zip(coordination,sentiment_face_detection):

            (start_x,start_y,end_x,end_y)=box

            sentiment=np.argmax(prediction)


            #determine the class label and color we will use to draw the bounding box and text
            label=resultat[sentiment]
            color=(0,255,0)
            #display the label and bounding boxes
            cv2.putText(frame,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

            cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),color,2)
        #show the output frame
        cv2.imshow("Frame",frame)
        key=cv2.waitKey(1) & 0xFF

        if key==ord('q'):
            break
    except:
        cv2.imshow("Frame",frame)
        key=cv2.waitKey(1) & 0xFF

        if key==ord('q'):
            break

cv2.destroyAllWindows()
video_stream.stream.release()


# In[ ]:
